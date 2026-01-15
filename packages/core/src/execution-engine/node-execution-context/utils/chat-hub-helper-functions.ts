import type {
	ChatHubProxyFunctions,
	INode,
	Workflow,
	IWorkflowExecuteAdditionalData,
} from 'n8n-workflow';

export function getChatHubHelperFunctions(
	additionalData: IWorkflowExecuteAdditionalData,
	workflow: Workflow,
	node: INode,
): Partial<ChatHubProxyFunctions> {
	const chatHubProxyProvider = additionalData['chat-hub']?.chatHubProxyProvider;
	if (!chatHubProxyProvider) return {};
	return {
		getChatHubProxy: (sessionId: string, memoryNodeId: string, turnId: string | null) =>
			chatHubProxyProvider.getChatHubProxy(
				workflow,
				node,
				sessionId,
				memoryNodeId,
				turnId,
				additionalData.userId,
			),
	};
}
