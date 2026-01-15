import { BaseChatMessageHistory } from '@langchain/core/chat_history';
import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type {
	IChatHubMemoryService,
	ChatHubMemoryEntry,
	StoredHumanMessage,
	StoredAIMessage,
	StoredSystemMessage,
	StoredToolMessage,
} from 'n8n-workflow';

/**
 * LangChain message history implementation that uses n8n's Chat Hub memory.
 * Memory is stored separately from chat UI messages, allowing:
 * - Multiple memory nodes in the same workflow to have isolated memory
 * - Proper branching on edit/retry via parentMessageId linking
 */
export class ChatHubMessageHistory extends BaseChatMessageHistory {
	lc_namespace = ['langchain', 'stores', 'message', 'n8n_chat_hub'];

	private memoryService: IChatHubMemoryService;

	constructor(options: { memoryService: IChatHubMemoryService }) {
		super();
		this.memoryService = options.memoryService;
	}

	async getMessages(): Promise<BaseMessage[]> {
		const entries = await this.memoryService.getMemory();
		return entries.map((entry) => this.convertToLangChainMessage(entry));
	}

	private convertToLangChainMessage(entry: ChatHubMemoryEntry): BaseMessage {
		switch (entry.role) {
			case 'human': {
				return this.asHumanMessage(entry);
			}

			case 'ai': {
				return this.asAIMessage(entry);
			}

			case 'tool': {
				return this.asToolMessage(entry);
			}

			case 'system': {
				return this.asSystemMessage(entry);
			}

			default: {
				// Unknown role treated as system
				return this.asSystemMessage(entry);
			}
		}
	}

	private isHumanMessage(content: unknown): content is StoredHumanMessage {
		return (
			typeof content === 'object' &&
			content !== null &&
			'content' in content &&
			typeof content.content === 'string'
		);
	}

	private isAIMessage(content: unknown): content is StoredAIMessage {
		return (
			typeof content === 'object' &&
			content !== null &&
			'content' in content &&
			typeof content.content === 'string' &&
			'toolCalls' in content &&
			Array.isArray(content.toolCalls)
		);
	}

	private isToolMessage(content: unknown): content is StoredToolMessage {
		return (
			typeof content === 'object' &&
			content !== null &&
			'toolCallId' in content &&
			'toolName' in content &&
			'toolInput' in content &&
			'toolOutput' in content
		);
	}

	private isSystemMessage(content: unknown): content is StoredSystemMessage {
		return (
			typeof content === 'object' &&
			content !== null &&
			'content' in content &&
			typeof content.content === 'string'
		);
	}

	private asHumanMessage(entry: ChatHubMemoryEntry): HumanMessage {
		if (this.isHumanMessage(entry.content)) {
			const humanData = entry.content;
			return new HumanMessage({ content: humanData.content, name: undefined });
		} else {
			return new HumanMessage({
				content: JSON.stringify(entry.content),
				name: undefined,
			});
		}
	}

	private asAIMessage(entry: ChatHubMemoryEntry): AIMessage {
		if (this.isAIMessage(entry.content)) {
			const aiData = entry.content;
			return new AIMessage({
				content: aiData.content,
				tool_calls: aiData.toolCalls,
				name: undefined,
			});
		} else {
			return new AIMessage({
				content: JSON.stringify(entry.content),
				name: undefined,
			});
		}
	}

	private asToolMessage(entry: ChatHubMemoryEntry): ToolMessage {
		if (this.isToolMessage(entry.content)) {
			const toolData = entry.content;
			return new ToolMessage({
				content: JSON.stringify(toolData.toolOutput),
				tool_call_id: toolData.toolCallId,
				name: toolData.toolName,
			});
		} else {
			return new ToolMessage({
				content: JSON.stringify(entry.content),
				tool_call_id: 'unknown',
				name: 'unknown',
			});
		}
	}

	private asSystemMessage(entry: ChatHubMemoryEntry): SystemMessage {
		if (this.isSystemMessage(entry.content)) {
			const systemData = entry.content;
			return new SystemMessage({ content: systemData.content });
		} else {
			return new SystemMessage({ content: JSON.stringify(entry.content) });
		}
	}

	async addMessage(message: BaseMessage): Promise<void> {
		const content =
			typeof message.content === 'string' ? message.content : JSON.stringify(message.content);

		if (message.type === 'human' && message instanceof HumanMessage) {
			await this.memoryService.addHumanMessage(content);
		} else if (message.type === 'ai' && message instanceof AIMessage) {
			const toolCalls = message.tool_calls ?? [];
			await this.memoryService.addAIMessage(content, toolCalls);
		} else if (message.type === 'tool' && message instanceof ToolMessage) {
			const toolMessage = message as ToolMessage;
			await this.memoryService.addToolMessage(
				toolMessage.tool_call_id,
				toolMessage.name ?? 'unknown',
				{}, // Input not available from ToolMessage
				typeof toolMessage.content === 'string' ? toolMessage.content : toolMessage.content,
			);
		}
		// System messages are not saved in conversation history
	}

	async addMessages(messages: BaseMessage[]): Promise<void> {
		for (const message of messages) {
			await this.addMessage(message);
		}
	}

	async addUserMessage(message: string): Promise<void> {
		await this.addMessage(new HumanMessage(message));
	}

	async addAIMessage(message: string): Promise<void> {
		await this.addMessage(new AIMessage(message));
	}

	async clear(): Promise<void> {
		await this.memoryService.clearMemory();
	}
}
