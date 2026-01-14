import { Logger } from '@n8n/backend-common';
import { Service } from '@n8n/di';
import {
	ChatHubProxyProvider,
	IChatHubMemoryService,
	ChatHubMemoryEntry,
	INode,
	Workflow,
} from 'n8n-workflow';
import { v4 as uuid } from 'uuid';

import { buildMessageHistory, extractTurnIds } from './chat-hub-history.utils';
import { ChatHubMemory } from './chat-hub-memory.entity';
import { ChatHubMemoryRepository } from './chat-hub-memory.repository';
import { ChatHubMessageRepository } from './chat-message.repository';
import { ChatHubSessionRepository } from './chat-session.repository';

const ALLOWED_NODES = ['@n8n/n8n-nodes-langchain.memoryChatHub'] as const;
const NAME_FALLBACK = 'Workflow Chat';

type AllowedNode = (typeof ALLOWED_NODES)[number];

export function isAllowedNode(s: string): s is AllowedNode {
	return ALLOWED_NODES.includes(s as AllowedNode);
}

@Service()
export class ChatHubProxyService implements ChatHubProxyProvider {
	constructor(
		private readonly memoryRepository: ChatHubMemoryRepository,
		private readonly messageRepository: ChatHubMessageRepository,
		private readonly sessionRepository: ChatHubSessionRepository,
		private readonly logger: Logger,
	) {
		this.logger = this.logger.scoped('chat-hub');
	}

	private validateRequest(node: INode) {
		if (!isAllowedNode(node.type)) {
			throw new Error('This proxy is only available for Chat Hub Memory nodes');
		}
	}

	async getChatHubProxy(
		workflow: Workflow,
		node: INode,
		sessionId: string,
		memoryNodeId: string,
		turnId: string | null,
		ownerId?: string,
	): Promise<IChatHubMemoryService> {
		this.validateRequest(node);

		if (!ownerId) {
			throw new Error(
				'Owner ID is required for Chat Hub Memory. For manual executions, ensure the user context is available.',
			);
		}

		// Extract workflow info for session creation
		const workflowId = workflow.id;
		const agentName = this.extractAgentName(workflow);

		return this.makeChatHubOperations(
			sessionId,
			memoryNodeId,
			turnId,
			ownerId,
			workflowId,
			agentName,
		);
	}

	/**
	 * Extract agent name from the chat trigger node's agentName parameter,
	 * falling back to workflow name if not set.
	 */
	private extractAgentName(workflow: Workflow): string {
		// Look for chat trigger node
		const chatTriggerNode = Object.values(workflow.nodes).find(
			(n) => n.type === '@n8n/n8n-nodes-langchain.chatTrigger',
		);

		if (
			typeof chatTriggerNode?.parameters?.agentName === 'string' &&
			chatTriggerNode.parameters.agentName.trim() !== ''
		) {
			return String(chatTriggerNode.parameters.agentName);
		}

		// Fall back to workflow name or default
		if (workflow.name && workflow.name.trim() !== '') {
			return workflow.name;
		}

		return NAME_FALLBACK;
	}

	private makeChatHubOperations(
		sessionId: string,
		memoryNodeId: string,
		providedTurnId: string | null,
		ownerId: string,
		workflowId: string | undefined,
		agentName: string,
	): IChatHubMemoryService {
		const memoryRepository = this.memoryRepository;
		const messageRepository = this.messageRepository;
		const sessionRepository = this.sessionRepository;
		const logger = this.logger;

		// turnId is a correlation ID generated BEFORE workflow execution starts.
		// It links memory entries created during this execution to the AI message that will be saved later.
		// For manual executions (turnId is null), we generate a random one to enable basic linear history.
		const turnId = providedTurnId ?? uuid();

		return {
			getOwnerId() {
				return ownerId;
			},

			async getMemory(): Promise<ChatHubMemoryEntry[]> {
				let memoryEntries: ChatHubMemory[];

				if (providedTurnId === null) {
					// Manual execution: load ALL memory for this session+node (simple linear history)
					logger.debug('Loading all memory for node (manual execution)', {
						sessionId,
						memoryNodeId,
					});
					memoryEntries = await memoryRepository.getAllMemoryForNode(sessionId, memoryNodeId);
				} else {
					// Chat Hub execution: use turnId-based filtering for edit/retry branching
					const chatMessages = await messageRepository.getManyBySessionId(sessionId);

					if (chatMessages.length === 0) {
						return [];
					}

					// Build the message chain - this automatically excludes superseded messages
					// (those that have been replaced by edits or retries)
					const messageChain = buildMessageHistory(chatMessages);

					// Extract turn IDs from AI messages in the chain
					// Memory entries are linked by turnId, so we load memory
					// for all non-superseded AI messages in the conversation
					const turnIds = extractTurnIds(messageChain);

					if (turnIds.length === 0) {
						// No AI messages yet (first message in conversation)
						return [];
					}

					logger.debug('Loading memory for turns in chain', {
						sessionId,
						memoryNodeId,
						turnIds,
					});

					memoryEntries = await memoryRepository.getMemoryByTurnIds(
						sessionId,
						memoryNodeId,
						turnIds,
					);
				}

				return memoryEntries.map((entry) => ({
					id: entry.id,
					role: entry.role,
					content: entry.content,
					name: entry.name,
					createdAt: entry.createdAt,
				}));
			},

			async addHumanMessage(content: string): Promise<void> {
				const id = uuid();
				await memoryRepository.createMemoryEntry({
					id,
					sessionId,
					memoryNodeId,
					turnId,
					role: 'human',
					content,
					name: 'User',
				});
				logger.debug('Added human message to memory', {
					sessionId,
					memoryNodeId,
					memoryId: id,
					turnId,
				});
			},

			async addAIMessage(content: string): Promise<void> {
				const id = uuid();
				await memoryRepository.createMemoryEntry({
					id,
					sessionId,
					memoryNodeId,
					turnId,
					role: 'ai',
					content,
					name: 'AI',
				});
				logger.debug('Added AI message to memory', {
					sessionId,
					memoryNodeId,
					memoryId: id,
					turnId,
				});
			},

			async addToolMessage(
				toolCallId: string,
				toolName: string,
				toolInput: unknown,
				toolOutput: unknown,
			): Promise<void> {
				const id = uuid();
				const content = JSON.stringify({
					toolCallId,
					toolName,
					toolInput,
					toolOutput,
				});

				await memoryRepository.createMemoryEntry({
					id,
					sessionId,
					memoryNodeId,
					turnId,
					role: 'tool',
					content,
					name: toolName,
				});
				logger.debug('Added tool message to memory', {
					sessionId,
					memoryNodeId,
					memoryId: id,
					toolName,
					turnId,
				});
			},

			async clearMemory(): Promise<void> {
				await memoryRepository.deleteBySessionAndNode(sessionId, memoryNodeId);
				logger.debug('Cleared memory for node', { sessionId, memoryNodeId });
			},

			async ensureSession(): Promise<void> {
				const exists = await sessionRepository.existsById(sessionId, ownerId);
				if (!exists) {
					const sessionTitle = agentName;
					await sessionRepository.createChatSession({
						id: sessionId,
						ownerId,
						title: sessionTitle,
						lastMessageAt: new Date(),
						tools: [],
						provider: 'n8n',
						credentialId: null,
						model: null,
						workflowId: workflowId ?? null,
						agentId: null,
						agentName,
					});
					logger.debug('Created new chat hub session', {
						sessionId,
						ownerId,
						title: sessionTitle,
						workflowId,
						agentName,
					});
				}
			},
		};
	}
}
