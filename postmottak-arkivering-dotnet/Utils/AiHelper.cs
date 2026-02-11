using System;
using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.MistralAI;
using Microsoft.SemanticKernel.Connectors.MistralAI.Client;
using OpenAI.Chat;
using postmottak_arkivering_dotnet.Contracts.Ai.ChatResult;
using Serilog;
using Serilog.Context;
using ChatMessageContent = Microsoft.SemanticKernel.ChatMessageContent;

namespace postmottak_arkivering_dotnet.Utils;

public enum AiProvider
{
    AzureOpenAI,
    Mistral
}

internal static class AiHelper
{
    internal static IConfigurationManager? ConfigurationManager { get; set; }

    internal static IKernelBuilder CreateNewKernelBuilder(LogLevel logLevel)
    {
        if (ConfigurationManager is null)
        {
            throw new NullReferenceException("ConfigurationManager is not set");
        }

        var provider = GetAiProvider();
        var kernelBuilder = Kernel.CreateBuilder();

        switch (provider)
        {
            case AiProvider.AzureOpenAI:
                AddAzureOpenAIProvider(kernelBuilder);
                break;
            case AiProvider.Mistral:
                AddMistralProvider(kernelBuilder);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported AI provider");
        }

        kernelBuilder.Services.AddLogging(configure =>
        {
            configure
                .AddConsole()
                .SetMinimumLevel(logLevel);
        });
        
        return kernelBuilder;
    }

    private static AiProvider GetAiProvider()
    {
        var providerName = ConfigurationManager!["AI_PROVIDER"] ?? "AzureOpenAI";
        return Enum.TryParse<AiProvider>(providerName, true, out var provider) 
            ? provider 
            : AiProvider.AzureOpenAI;
    }

    private static void AddAzureOpenAIProvider(IKernelBuilder kernelBuilder)
    {
        var modelName = ConfigurationManager!["AZURE_OPENAI_MODEL_NAME"] ?? throw new NullReferenceException("AZURE_OPENAI_MODEL_NAME is required for Azure OpenAI provider");
        var apiKey = ConfigurationManager["AZURE_OPENAI_API_KEY"] ?? throw new NullReferenceException("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider");
        var endpoint = ConfigurationManager["AZURE_OPENAI_ENDPOINT"] ?? throw new NullReferenceException("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI provider");

        kernelBuilder.AddAzureOpenAIChatCompletion(modelName, endpoint, apiKey);
    }

    private static void AddMistralProvider(IKernelBuilder kernelBuilder)
    {
        var modelName = ConfigurationManager!["MISTRAL_MODEL_NAME"] ?? "mistral-large-latest";
        var apiKey = ConfigurationManager["MISTRAL_API_KEY"] ?? throw new NullReferenceException("MISTRAL_API_KEY is required for Mistral provider");
        
        // Use the official Mistral AI connector
        kernelBuilder.AddMistralChatCompletion(
            modelId: modelName,
            apiKey: apiKey
        );
    }

    internal static ChatCompletionAgent CreateNewAgent(IKernelBuilder kernelBuilder, string agentName, string agentInstructions, Type responseFormat)
    {
        if (ConfigurationManager is null)
        {
            throw new NullReferenceException("ConfigurationManager is not set");
        }
        
        var provider = GetAiProvider();
        int maxCompletionTokens = GetMaxCompletionTokens(provider);
        
        Kernel kernel = kernelBuilder.Build();
        
        var promptExecutionSettings = CreatePromptExecutionSettings(provider, maxCompletionTokens, responseFormat);
        
        // Enhance instructions for both providers to avoid markdown formatting
        var finalInstructions = EnhanceInstructionsForJsonResponse(agentInstructions, responseFormat, provider);
        
        return new ChatCompletionAgent
        {
            Name = agentName,
            Instructions = finalInstructions,
            Kernel = kernel,
            Arguments = new KernelArguments(promptExecutionSettings)
        };
    }

    private static int GetMaxCompletionTokens(AiProvider provider)
    {
        var configKey = provider switch
        {
            AiProvider.AzureOpenAI => "AZURE_OPENAI_MAX_COMPLETION_TOKENS",
            AiProvider.Mistral => "MISTRAL_MAX_COMPLETION_TOKENS",
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported AI provider")
        };

        return int.TryParse(ConfigurationManager![configKey], out int maxTokens) ? maxTokens : 10000;
    }

    private static PromptExecutionSettings CreatePromptExecutionSettings(AiProvider provider, int maxTokens, Type responseFormat)
    {
        return provider switch
        {
            AiProvider.AzureOpenAI => new AzureOpenAIPromptExecutionSettings
            {
                MaxTokens = maxTokens,
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
                Store = false,
                ResponseFormat = responseFormat,
            },
            AiProvider.Mistral => new MistralAIPromptExecutionSettings
            {
                MaxTokens = maxTokens,
                Temperature = 0.7,
                // For JSON response format, let the instructions handle JSON format
                // Mistral may have different API structure than OpenAI
            },
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported AI provider")
        };
    }
    
    internal static T? GetLatestAnswer<T>(ChatHistory chatHistory)
    {
        // 0 will be the user input. 1 will be the AI response, and so on...
        if (chatHistory.Count < 2)
        {
            return default;
        }
        
        var content = chatHistory[^1].Content;
        if (string.IsNullOrEmpty(content))
        {
            return default;
        }

        try
        {
            // Clean the JSON content to remove markdown code blocks and extra text
            var cleanedContent = CleanJsonContent(content);
            
            // Use JsonSerializer options for better compatibility with different providers
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = false
            };
            
            var result = JsonSerializer.Deserialize<T>(cleanedContent, options) ?? throw new InvalidOperationException($"Failed to deserialize AI response into type {typeof(T).Name}");
            
            return result;
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Failed to deserialize AI response into type {typeof(T).Name}. Raw content: '{content}'. Error: {ex.Message}", ex);
        }
    }

    private static string CleanJsonContent(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return content;
        }

        Log.Logger.Information("Cleaning JSON content from AI response. Content: {Content}", content);
        // Remove markdown code blocks
        if (content.Contains("```"))
        {
            // Handle ```json blocks
            var jsonBlockStart = content.IndexOf("```json", StringComparison.OrdinalIgnoreCase);
            if (jsonBlockStart >= 0)
            {
                var blockStart = content.IndexOf('\n', jsonBlockStart) + 1;
                var blockEnd = content.IndexOf("```", blockStart);
                if (blockEnd > blockStart)
                {
                    content = content.Substring(blockStart, blockEnd - blockStart).Trim();
                }
            }
            // Handle generic ``` blocks
            else if (content.StartsWith("```"))
            {
                var firstNewline = content.IndexOf('\n');
                var lastBackticks = content.LastIndexOf("```");
                if (firstNewline >= 0 && lastBackticks > firstNewline)
                {
                    content = content.Substring(firstNewline + 1, lastBackticks - firstNewline - 1).Trim();
                }
            }
        }

        Log.Logger.Information("After removing code blocks, content: {Content}", content);

        // Find and extract JSON object if there's surrounding text
        var objectStart = content.IndexOf('{');
        var objectEnd = content.LastIndexOf('}');
        
        if (objectStart >= 0 && objectEnd > objectStart)
        {
            content = content.Substring(objectStart, objectEnd - objectStart + 1);
        }

        return content.Trim();
    }
    
    [SuppressMessage("ReSharper", "StructuredMessageTemplateProblem")]
    internal static async Task<ChatHistory> InvokeAgent(this ChatCompletionAgent agent, string prompt, string responseType, ChatHistory? chatHistory = null)
    {
        using (GlobalLogContext.PushProperty("AgentName", agent.Name))
        using (GlobalLogContext.PushProperty("ResponseType", responseType))
        {
            Log.Logger.Information("Asking {AgentName} for response type {ResponseType}");
            Log.Logger.Debug("{Prompt}", prompt);

            var history = chatHistory ?? new ChatHistory();

            AgentThread agentThread = new ChatHistoryAgentThread(history);
            Log.Logger.Debug("Invoking agent {AgentName}", agent.Name);
            
            var userMessage = new ChatMessageContent(AuthorRole.User, prompt);
            var responses = agent.InvokeAsync(userMessage, agentThread);
            
            await foreach (ChatMessageContent response in responses)
            {
                var resultContent = response.Content ?? string.Empty;

                var (inputTokens, outputTokens) = GetTokenUsage(response);
                Log.Logger.Information("Got {ResponseType} response from {AgentName}. InputTokenCount: {InputTokenCount}. OutputTokenCount: {OutputTokenCount}",
                    responseType, agent.Name, inputTokens, outputTokens);
                Log.Logger.Debug("{Result}", resultContent);
            }

            return history;
        }
    }

    private static (int? inputTokens, int? outputTokens) GetTokenUsage(ChatMessageContent response)
    {
        if (response.Metadata?.TryGetValue("Usage", out var usageObj) == true)
        {
            return usageObj switch
            {
                ChatTokenUsage openAiUsage => (openAiUsage.InputTokenCount, openAiUsage.OutputTokenCount),
                MistralUsage mistralUsage => (mistralUsage.PromptTokens, mistralUsage.CompletionTokens),
                _ => (null, null)
            };
        }
        
        return (null, null);
    }
    
    internal static string GetCurrentProviderInfo()
    {
        if (ConfigurationManager is null)
        {
            return "ConfigurationManager not initialized";
        }

        var provider = GetAiProvider();
        return provider switch
        {
            AiProvider.AzureOpenAI => $"Azure OpenAI - Model: {ConfigurationManager["AZURE_OPENAI_MODEL_NAME"] ?? "Not configured"}",
            AiProvider.Mistral => $"Mistral AI - Model: {ConfigurationManager["MISTRAL_MODEL_NAME"] ?? "mistral-large-latest"}",
            _ => "Unknown provider"
        };
    }

    private static string EnhanceInstructionsForJsonResponse(string originalInstructions, Type responseFormat, AiProvider provider)
    {
        if (responseFormat == typeof(object))
        {
            return originalInstructions;
        }

        var jsonInstructions = provider switch
        {
            AiProvider.Mistral => @"
IMPORTANT FOR MISTRAL - JSON RESPONSE RULES:
- Respond ONLY with valid JSON
- Do NOT use markdown code blocks (```json)
- Do NOT include explanatory text before or after the JSON
- Use proper JSON syntax with double quotes
- Return the JSON object directly without any formatting

Example: {""property"": ""value"", ""boolProperty"": true}",
            
            AiProvider.AzureOpenAI => @"
IMPORTANT - JSON RESPONSE RULES:
- Respond ONLY with valid JSON
- Do NOT use markdown code blocks (```json)
- Do NOT include explanatory text before or after the JSON
- Use proper JSON syntax with double quotes
- Return the JSON object directly without any formatting

Example: {""property"": ""value"", ""boolProperty"": true}",
            
            _ => ""
        };

        if (responseFormat.Name == "PengetransportenChatResult")
        {
            jsonInstructions += @"

SPECIFIC INSTRUCTIONS FOR PENGETRANSPORTEN:
You must analyze the email content and return a JSON with exactly these two properties:
- ""description"": A string describing what type of invoice/bill this is (e.g., ""Faktura"", ""Regning"", ""Purring"", ""Inkassovarsel"")
- ""isInvoiceRelated"": A boolean (true/false) - set to true ONLY if you are at least 90% certain this is invoice-related

Required JSON format:
{""description"": ""Faktura"", ""isInvoiceRelated"": true}

NEVER return empty description - always provide a classification like ""Faktura"", ""Regning"", ""Purring"", etc.";
        }

        if (responseFormat.Name == "Rf1350ChatResult")
        {
            jsonInstructions += @"

SPECIFIC INSTRUCTIONS FOR RF1350:
You must analyze the RF13.50 email content and extract the following information:
- ""Type"": The document type (e.g., ""Overføring av mottatt søknad"", ""Automatisk kvittering på innsendt søknad"", ""Anmodning om utbetaling"")
- ""ReferenceNumber"": The reference number in format YYYY-NNNN (e.g., ""2024-1234"")
- ""ProjectNumber"": The project number in format YY-NNNNNN (e.g., ""24-123456"")
- ""ProjectName"": The name/title of the project
- ""ProjectOwner"": The name of the project owner/responsible person
- ""OrganizationNumber"": The organization number (9 digits)

Required JSON format:
{""Type"": ""Overføring av mottatt søknad"", ""ReferenceNumber"": ""2024-1234"", ""ProjectNumber"": ""24-123456"", ""ProjectName"": ""Project Name"", ""ProjectOwner"": ""Owner Name"", ""OrganizationNumber"": 123456789}

IMPORTANT: 
- NEVER return empty strings or 0 values unless the information is genuinely not present in the email
- Look carefully for reference numbers, project numbers, organization names/numbers in the email content
- If you cannot find specific information, analyze the context and provide your best estimate based on the email content";
        }

        if (responseFormat.Name == "LoyvegarantiChatResult")
        {
            jsonInstructions += @"

SPECIFIC INSTRUCTIONS FOR LOYVEGARANTI:
You must analyze the email subject and content to extract information about insurance guarantees (løyvegaranti):
- ""Description"": A brief description of the content/purpose of the guarantee
- ""OrganizationName"": The name of the organization - ALWAYS in CAPITAL LETTERS, appears before 'Org.nr'
- ""OrganizationNumber"": The 9-digit organization number (may contain spaces)
- ""Type"": The type of guarantee operation - must be one of: ""Løyvegaranti"", ""EndringAvLøyvegaranti"", or ""OpphørAvLøyvegaranti""

Required JSON format:
{""Description"": ""Brief description"", ""OrganizationName"": ""ORGANIZATION NAME"", ""OrganizationNumber"": ""123 456 789"", ""Type"": ""Løyvegaranti""}

IMPORTANT:
- Organization name must be in CAPITAL LETTERS
- Organization number is exactly 9 digits (may have spaces)
- Type must be exactly one of the three enum values
- Look for keywords like 'løyve', 'garanti', 'endring', 'opphør' to determine the Type
- NEVER return empty strings - analyze the email content carefully";
        }

        return originalInstructions + jsonInstructions;
    }
}