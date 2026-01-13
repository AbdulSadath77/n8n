import { BaseChatMessageHistory } from '@langchain/core/chat_history';
import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { IChatHubSessionService, ChatHubMemoryMessage } from 'n8n-workflow';

/**
 * LangChain message history implementation that uses n8n's Chat Hub database.
 * This enables workflow agents to maintain persistent conversation state
 * that integrates with the Chat Hub UI.
 */
export class ChatHubMessageHistory extends BaseChatMessageHistory {
	lc_namespace = ['langchain', 'stores', 'message', 'n8n_chat_hub'];

	private proxy: IChatHubSessionService;

	private lastMessageId: string | null = null;

	private executionId?: number;

	constructor(options: { proxy: IChatHubSessionService; executionId?: number }) {
		super();
		this.proxy = options.proxy;
		this.executionId = options.executionId;
	}

	async getMessages(): Promise<BaseMessage[]> {
		const messages = await this.proxy.getMessages(this.lastMessageId ?? undefined);

		// Update lastMessageId to the last message in the chain for proper chaining
		// This ensures new messages chain off existing conversation instead of using null
		if (messages.length > 0) {
			this.lastMessageId = messages[messages.length - 1].id;
		}

		return messages.map((msg) => this.convertToLangChainMessage(msg));
	}

	private convertToLangChainMessage(msg: ChatHubMemoryMessage): BaseMessage {
		switch (msg.type) {
			case 'human':
				return new HumanMessage({ content: msg.content, name: msg.name });

			case 'ai':
				return new AIMessage({ content: msg.content, name: msg.name });

			case 'system':
				return new SystemMessage({ content: msg.content });

			case 'tool': {
				// Parse tool message content
				const toolData = this.parseToolMessageContent(msg.content);
				return new ToolMessage({
					content: JSON.stringify(toolData.toolOutput),
					tool_call_id: toolData.toolCallId,
					name: toolData.toolName,
				});
			}

			default:
				// Generic messages treated as system
				return new SystemMessage({ content: msg.content });
		}
	}

	private parseToolMessageContent(content: string): {
		toolCallId: string;
		toolName: string;
		toolInput: unknown;
		toolOutput: unknown;
	} {
		try {
			return JSON.parse(content);
		} catch {
			// Fallback for malformed tool messages
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

		if (messageType === 'human') {
			this.lastMessageId = await this.proxy.addHumanMessage(
				typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
				this.lastMessageId,
			);
		} else if (messageType === 'ai') {
			this.lastMessageId = await this.proxy.addAIMessage(
				typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
				this.lastMessageId,
				{ executionId: this.executionId },
			);
		} else if (messageType === 'tool') {
			const toolMsg = message as ToolMessage;
			this.lastMessageId = await this.proxy.addToolMessage(
				toolMsg.tool_call_id,
				toolMsg.name ?? 'unknown',
				{}, // Input not available from ToolMessage
				typeof toolMsg.content === 'string' ? toolMsg.content : toolMsg.content,
				this.lastMessageId,
				{ executionId: this.executionId },
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
		await this.proxy.clearMessages();
		this.lastMessageId = null;
	}
}
