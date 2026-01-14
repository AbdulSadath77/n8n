import { BaseChatMessageHistory } from '@langchain/core/chat_history';
import type { BaseMessage, ToolCall } from '@langchain/core/messages';
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { IChatHubMemoryService, ChatHubMemoryEntry } from 'n8n-workflow';

/**
 * Structure for storing human messages as JSON.
 */
interface StoredHumanMessage {
	content: string;
}

/**
 * Structure for storing AI messages as JSON.
 * Includes tool_calls array so ToolMessages can be properly matched when reconstructing history.
 */
interface StoredAIMessage {
	content: string;
	toolCalls: ToolCall[];
}

/**
 * Structure for storing tool messages as JSON.
 */
interface StoredToolMessage {
	toolCallId: string;
	toolName: string;
	toolInput: unknown;
	toolOutput: unknown;
}

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
				const humanData = this.parseHumanMessageContent(entry.content);
				return new HumanMessage({ content: humanData.content, name: entry.name ?? undefined });
			}

			case 'ai': {
				const aiData = this.parseAIMessageContent(entry.content);
				return new AIMessage({
					content: aiData.content,
					name: entry.name ?? undefined,
					tool_calls: aiData.toolCalls,
				});
			}

			case 'system':
				return new SystemMessage({ content: entry.content });

			case 'tool': {
				// Parse tool message content
				const toolData = this.parseToolMessageContent(entry.content);
				return new ToolMessage({
					content: JSON.stringify(toolData.toolOutput),
					tool_call_id: toolData.toolCallId,
					name: toolData.toolName,
				});
			}

			default:
				// Unknown role treated as system
				return new SystemMessage({ content: entry.content });
		}
	}

	/**
	 * Parse human message content stored as JSON: { content: "..." }
	 */
	private parseHumanMessageContent(content: string): { content: string } {
		try {
			const parsed = JSON.parse(content) as StoredHumanMessage;
			return {
				content:
					typeof parsed.content === 'string' ? parsed.content : JSON.stringify(parsed.content),
			};
		} catch {
			// Fallback for malformed data
			return { content };
		}
	}

	/**
	 * Parse AI message content stored as JSON: { content: "...", toolCalls: [...] }
	 */
	private parseAIMessageContent(content: string): { content: string; toolCalls: ToolCall[] } {
		try {
			const parsed = JSON.parse(content) as StoredAIMessage;
			return {
				content:
					typeof parsed.content === 'string' ? parsed.content : JSON.stringify(parsed.content),
				toolCalls: parsed.toolCalls ?? [],
			};
		} catch {
			// Fallback for malformed data
			return { content, toolCalls: [] };
		}
	}

	/**
	 * Parse tool message content stored as JSON.
	 */
	private parseToolMessageContent(content: string): StoredToolMessage {
		try {
			return JSON.parse(content) as StoredToolMessage;
		} catch {
			// Fallback for malformed data
			return {
				toolCallId: 'unknown',
				toolName: 'unknown',
				toolInput: {},
				toolOutput: content,
			};
		}
	}

	async addMessage(message: BaseMessage): Promise<void> {
		const messageType = message._getType();
		const content =
			typeof message.content === 'string' ? message.content : JSON.stringify(message.content);

		if (messageType === 'human') {
			const storedContent: StoredHumanMessage = { content };
			await this.memoryService.addHumanMessage(JSON.stringify(storedContent));
		} else if (messageType === 'ai') {
			const aiMsg = message as AIMessage;
			const storedContent: StoredAIMessage = {
				content,
				toolCalls: aiMsg.tool_calls ?? [],
			};
			await this.memoryService.addAIMessage(JSON.stringify(storedContent));
		} else if (messageType === 'tool') {
			const toolMsg = message as ToolMessage;
			await this.memoryService.addToolMessage(
				toolMsg.tool_call_id,
				toolMsg.name ?? 'unknown',
				{}, // Input not available from ToolMessage
				typeof toolMsg.content === 'string' ? toolMsg.content : toolMsg.content,
			);
		}
		// System messages are typically not saved in conversation history
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
