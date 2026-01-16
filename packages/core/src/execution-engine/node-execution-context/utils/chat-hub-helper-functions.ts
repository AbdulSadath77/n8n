import type {
	ChatHubProxyFunctions,
	INode,
	Workflow,
	IWorkflowExecuteAdditionalData,
	WorkflowExecuteMode,
} from 'n8n-workflow';

export function getChatHubHelperFunctions(
	additionalData: IWorkflowExecuteAdditionalData,
	workflow: Workflow,
	node: INode,
	mode: WorkflowExecuteMode,
): Partial<ChatHubProxyFunctions> {
	const chatHubProxyProvider = additionalData['chat-hub']?.chatHubProxyProvider;
	if (!chatHubProxyProvider) return {};
	return {
		getChatHubProxy: (
			sessionId: string,
			memoryNodeId: string,
			turnId: string | null,
			previousTurnIds: string[],
		) =>
			chatHubProxyProvider.getChatHubProxy(
				workflow,
				node,
				sessionId,
				memoryNodeId,
				turnId,
				previousTurnIds,
				mode === 'manual' ? undefined : additionalData.userId,
			),
	};
}
