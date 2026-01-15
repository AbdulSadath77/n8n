import { BufferWindowMemory } from '@langchain/classic/memory';
import { getSessionId } from '@utils/helpers';
import { logWrapper } from '@utils/logWrapper';
import { getConnectionHintNoticeField } from '@utils/sharedFields';
import {
	NodeConnectionTypes,
	NodeOperationError,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import {
	sessionIdOption,
	sessionKeyProperty,
	contextWindowLengthProperty,
	expressionSessionKeyProperty,
} from '../descriptions';
import { ChatHubMessageHistory } from './ChatHubMessageHistory';

export class MemoryChatHub implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Chat Hub Memory',
		name: 'memoryChatHub',
		icon: 'fa:comments',
		iconColor: 'blue',
		group: ['transform'],
		version: 1,
		description: 'Stores chat memory in n8n Chat Hub for persistent conversations',
		defaults: {
			name: 'Chat Hub Memory',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Memory'],
				Memory: ['For beginners'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.memorychathub/',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiMemory],
		outputNames: ['Memory'],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiAgent]),
			{
				displayName:
					"This memory stores conversations in n8n's local database, enabling simple persistent chat memory with support for Chat Hub's message edits and retries.",
				name: 'chatHubNotice',
				type: 'notice',
				default: '',
			},
			sessionIdOption,
			expressionSessionKeyProperty(1),
			sessionKeyProperty,
			contextWindowLengthProperty,
			{
				// Hidden parameter for turnId - injected by Chat Hub service before workflow execution.
				// This is a correlation ID generated BEFORE the workflow runs, linking memory entries
				// to the AI message that will be created for this execution turn.
				// On regeneration, a new turnId is generated, so old memory is automatically excluded.
				displayName: 'Turn ID',
				name: 'turnId',
				type: 'hidden',
				default: '',
				description: 'Correlation ID for this execution turn (set by Chat Hub)',
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const sessionId = getSessionId(this, itemIndex);
		const contextWindowLength = this.getNodeParameter('contextWindowLength', itemIndex) as number;
		const turnId = (this.getNodeParameter('turnId', itemIndex, '') as string) || null;

		// Get the node's internal ID to use as memoryNodeId
		const node = this.getNode();
		const memoryNodeId = node.id;

		// Get the Chat Hub proxy
		// turnId is a correlation ID generated before execution starts.
		// When provided by Chat Hub, it links memory entries to the AI message for edit/retry support.
		// When null (manual executions), the proxy generates a random one to enable basic linear history.
		const memoryService = this.helpers.getChatHubProxy?.(sessionId, memoryNodeId, turnId);

		if (!memoryService) {
			throw new NodeOperationError(
				node,
				'Chat Hub module is not available. Ensure the chat-hub module is enabled.',
			);
		}

		// Auto-create session if needed (manual executions)
		await memoryService.ensureSession();

		const chatHistory = new ChatHubMessageHistory({
			memoryService,
		});

		const memory = new BufferWindowMemory({
			k: contextWindowLength,
			memoryKey: 'chat_history',
			chatHistory,
			returnMessages: true,
			inputKey: 'input',
			outputKey: 'output',
		});

		return {
			response: logWrapper(memory, this),
		};
	}
}
