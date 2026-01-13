import { Logger } from '@n8n/backend-common';
import { Service } from '@n8n/di';
import { v4 as uuid } from 'uuid';
import {
	ChatHubProxyProvider,
	IChatHubSessionService,
	ChatHubMemoryMessage,
	AddAIMessageOptions,
	AddToolMessageOptions,
	INode,
	Workflow,
} from 'n8n-workflow';

import { buildMessageHistory } from './chat-hub-history.utils';
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
		ownerId?: string,
	): Promise<IChatHubSessionService> {
		this.validateRequest(node);

		if (!ownerId) {
			throw new Error(
				'Owner ID is required for Chat Hub Memory. For manual executions, ensure the user context is available.',
			);
		}

		// Extract workflow info for session creation
		const workflowId = workflow.id;
		const agentName = this.extractAgentName(workflow);

		return this.makeChatHubOperations(sessionId, ownerId, workflowId, agentName);
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

		if (chatTriggerNode?.parameters?.agentName) {
			return String(chatTriggerNode.parameters.agentName);
		}

		// Fall back to workflow name
		return workflow.name || NAME_FALLBACK;
	}

	private makeChatHubOperations(
		sessionId: string,
		ownerId: string,
		workflowId: string | undefined,
		agentName: string,
	): IChatHubSessionService {
		const messageRepository = this.messageRepository;
		const sessionRepository = this.sessionRepository;
		const logger = this.logger;

		return {
			getOwnerId() {
				return ownerId;
			},

			async getMessages(lastMessageId?: string): Promise<ChatHubMemoryMessage[]> {
				const messages = await messageRepository.getManyBySessionId(sessionId);

				if (messages.length === 0) {
					return [];
				}

				return buildMessageHistory(messages, lastMessageId);
			},

			async addHumanMessage(content: string, previousMessageId: string | null): Promise<string> {
				const id = uuid();
				await messageRepository.createChatMessage({
					id,
					sessionId,
					type: 'human',
					content,
					name: 'User',
					status: 'success',
					previousMessageId,
					provider: null,
					model: null,
					workflowId: null,
					agentId: null,
					executionId: null,
					retryOfMessageId: null,
					revisionOfMessageId: null,
					attachments: null,
				});
				logger.debug('Added human message to chat hub', { sessionId, messageId: id });
				return id;
			},

			async addAIMessage(
				content: string,
				previousMessageId: string | null,
				options: AddAIMessageOptions = {},
			): Promise<string> {
				const id = uuid();
				await messageRepository.createChatMessage({
					id,
					sessionId,
					type: 'ai',
					content,
					name: 'AI',
					status: 'success',
					previousMessageId,
					executionId: options.executionId ?? null,
					provider: (options.provider as 'n8n') ?? 'n8n',
					model: options.model ?? null,
					workflowId: null,
					agentId: null,
					retryOfMessageId: null,
					revisionOfMessageId: null,
					attachments: null,
				});
				logger.debug('Added AI message to chat hub', { sessionId, messageId: id });
				return id;
			},

			async addToolMessage(
				toolCallId: string,
				toolName: string,
				toolInput: unknown,
				toolOutput: unknown,
				previousMessageId: string | null,
				options: AddToolMessageOptions = {},
			): Promise<string> {
				const id = uuid();
				const content = JSON.stringify({
					toolCallId,
					toolName,
					toolInput,
					toolOutput,
				});

				await messageRepository.createChatMessage({
					id,
					sessionId,
					type: 'tool',
					content,
					name: toolName,
					status: 'success',
					previousMessageId,
					executionId: options.executionId ?? null,
					provider: null,
					model: null,
					workflowId: null,
					agentId: null,
					retryOfMessageId: null,
					revisionOfMessageId: null,
					attachments: null,
				});
				logger.debug('Added tool message to chat hub', {
					sessionId,
					messageId: id,
					toolName,
				});
				return id;
			},

			async clearMessages(): Promise<void> {
				await messageRepository.deleteBySessionId(sessionId);
				logger.debug('Cleared all messages for session', { sessionId });
			},

			async ensureSession(title?: string): Promise<void> {
				const exists = await sessionRepository.existsById(sessionId, ownerId);
				if (!exists) {
					const sessionTitle = title || NAME_FALLBACK;
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
