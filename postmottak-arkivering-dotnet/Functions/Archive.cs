using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.Storage.Blobs.Models;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Attributes;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Enums;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Graph.Models;
using Microsoft.OpenApi.Models;
using postmottak_arkivering_dotnet.Contracts;
using postmottak_arkivering_dotnet.Contracts.Ai;
using postmottak_arkivering_dotnet.Contracts.Ai.ChatResult;
using postmottak_arkivering_dotnet.Contracts.Email;
using postmottak_arkivering_dotnet.Services;
using postmottak_arkivering_dotnet.Services.Ai;
using postmottak_arkivering_dotnet.Utils;
using Serilog.Context;
using Vestfold.Extensions.Metrics.Services;
using FromBodyAttribute = Microsoft.Azure.Functions.Worker.Http.FromBodyAttribute;

namespace postmottak_arkivering_dotnet.Functions;

public class Archive
{
    private readonly IAiArntIvanService _aiArntIvanService;
    private readonly IAiPluginTestService _aiPluginTestService;
    private readonly IBlobService _blobService;
    private readonly IEmailTypeService _emailTypeService;
    private readonly IGraphService _graphService;
    private readonly ILogger<Archive> _logger;
    private readonly IMetricsService _metricsService;
    private readonly IStatisticsService _statisticsService;
    private readonly IHostEnvironment _hostEnvironment;
    private readonly IMongoDbServices _mongoDbServices;
    
    private readonly string _blobStorageFailedName;
    private readonly string _blobStorageQueueName;
    private readonly string _mailFolderInboxId;
    private readonly string _mailFolderManualHandlingId;
    private readonly string _mailFolderFinishedId;
    private readonly string _mailFolderUnwantedRuleId;
    private readonly string _mailFolderMaybeId;
    private readonly string _mailFolderNoMatchId;
    private readonly string _postboxUpn;
    private readonly string _postboxLogUpn;
    private readonly int[] _retryMinutesIntervals;

    public Archive(IConfiguration configuration,
        IAiArntIvanService aiArntIvanService,
        IAiPluginTestService aiPluginTestService,
        IBlobService blobService,
        IEmailTypeService emailTypeService,
        IGraphService graphService,
        ILogger<Archive> logger,
        IMetricsService metricsService,
        IStatisticsService statisticsService,
        IHostEnvironment hostEnvironment,
        IMongoDbServices mongoDbServices)
    {
        _aiArntIvanService = aiArntIvanService;
        _aiPluginTestService = aiPluginTestService;
        _blobService = blobService;
        _emailTypeService = emailTypeService;
        _graphService = graphService;
        _logger = logger;
        _statisticsService = statisticsService;
        _metricsService = metricsService;
        _hostEnvironment = hostEnvironment;
        _mongoDbServices = mongoDbServices;

        _blobStorageFailedName = configuration["BLOB_STORAGE_FAILED_NAME"] ?? "failed";
        _blobStorageQueueName = configuration["BLOB_STORAGE_QUEUE_NAME"] ?? "queue";
        _retryMinutesIntervals = configuration["RETRY_INTERVALS"]?.Split(',').Select(int.Parse).ToArray() ?? throw new NullReferenceException();
        
        _mailFolderInboxId = configuration["POSTMOTTAK_MAIL_FOLDER_INBOX_ID"] ?? throw new NullReferenceException();
        _mailFolderManualHandlingId = configuration["POSTMOTTAK_MAIL_FOLDER_MANUALHANDLING_ID"] ?? throw new NullReferenceException();
        _mailFolderFinishedId = configuration["POSTMOTTAK_MAIL_FOLDER_FINISHED_ID"] ?? throw new NullReferenceException();
        _mailFolderUnwantedRuleId = configuration["POSTMOTTAK_MAIL_FOLDER_UNWANTED_RULE_ID"] ?? throw new NullReferenceException();
        _mailFolderMaybeId = configuration["POSTMOTTAK_MAIL_FOLDER_ROBOT_LOG_MAYBE_ID"] ?? throw new NullReferenceException();
        _mailFolderNoMatchId = configuration["POSTMOTTAK_MAIL_FOLDER_ROBOT_LOG_NOMATCH_ID"] ?? throw new NullReferenceException();
        _postboxUpn = configuration["POSTMOTTAK_UPN"] ?? throw new NullReferenceException();
        _postboxLogUpn = configuration["POSTMOTTAK_LOG_UPN"] ?? throw new NullReferenceException();
    }

    [Function("ArchiveEmails")]
    [OpenApiOperation(operationId: "ArchiveEmails")]
    [OpenApiSecurity("Authentication", SecuritySchemeType.ApiKey, Name = "X-Functions-Key", In = OpenApiSecurityLocationType.Header)]
    [OpenApiResponseWithBody(HttpStatusCode.OK, contentType: "application/json", typeof(ArchiveOkResponse), Description = "Executed successfully")]
    public async Task<IActionResult> ArchiveEmails([HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequest req)
    {
        ArchiveOkResponse response = await GetAndHandleEmails();

        return new OkObjectResult(response);
    }
    
    [Function("GetAndHandleEmailsTimer")]
    public async Task GetAndHandleEmailsTrigger([TimerTrigger("%SynchronizeSchedule%")] TimerInfo myTimer)
    {
        if (_hostEnvironment.IsDevelopment())
        {
            _logger.LogInformation("Development environment detected, skipping GetAndHandleEmailsTimer email handling.");
            
            await Task.WhenAll(
                MovedByRulesCount(),
                BlobStorageFailedCount(),
                BlobStorageQueueCount()
            );

            return;
        }
        
        await Task.WhenAll(
            MovedByRulesCount(),
            BlobStorageFailedCount(),
            GetAndHandleEmails()
        );
    }
    
    [Function("ListFolders")]
    [OpenApiOperation(operationId: "ListFolders")]
    [OpenApiSecurity("Authentication", SecuritySchemeType.ApiKey, Name = "X-Functions-Key", In = OpenApiSecurityLocationType.Header)]
    [OpenApiResponseWithBody(HttpStatusCode.OK, "application/json", typeof(List<MailFolder>), Description = "List of mail folders")]
    [OpenApiResponseWithBody(HttpStatusCode.InternalServerError, "application/json", typeof(ErrorResponse), Description = "Error occured")]
    public async Task<IActionResult> ListFolders([HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequest req)
    {
        var parentFolder = req.Query.Keys.Contains("parentFolder") ? req.Query["parentFolder"].ToString() : null;
        
        var mailFolders = !string.IsNullOrEmpty(parentFolder)
            ? await _graphService.GetMailChildFolders(_postboxUpn, parentFolder)
            : await _graphService.GetMailFolders(_postboxUpn);

        return new OkObjectResult(mailFolders);
    }

    [Function("AskArntIvan")]
    [OpenApiOperation(operationId: "AskArntIvan")]
    [OpenApiSecurity("Authentication", SecuritySchemeType.ApiKey, Name = "X-Functions-Key",
        In = OpenApiSecurityLocationType.Header)]
    [OpenApiRequestBody("application/json", typeof(AiPromptRequest), Description = "Prompt to ask Arnt Ivan")]
    [OpenApiResponseWithBody(HttpStatusCode.OK, "text/plain", typeof(string), Description = "Response from Arnt Ivan")]
    [OpenApiResponseWithBody(HttpStatusCode.BadRequest, "text/plain", typeof(string),
        Description = "Unknown agent")]
    [OpenApiResponseWithBody(HttpStatusCode.InternalServerError, "text/plain", typeof(string),
        Description = "Error occured")]
    public async Task<IActionResult> AskArntIvan([HttpTrigger(AuthorizationLevel.Function, "post")] HttpRequest req,
        [FromBody] AiPromptRequest promptRequest)
    {
        _logger.LogInformation("AskArntIvan function started for {AgentName} with {Prompt}", promptRequest.Agent, promptRequest.Prompt);
        
        switch (promptRequest.Agent)
        {
            case "ArntIvan":
            {
                var (_, result) = await _aiArntIvanService.Ask<GeneralChatResult>(promptRequest.Prompt);
                return new OkObjectResult(result);
            }
            case "FunFact":
            {
                var result = await _aiArntIvanService.FunFact();
                return new OkObjectResult(result);
            }
            case "Pengetransporten":
            {
                var (_, result) = await _aiArntIvanService.Ask<PengetransportenChatResult>(promptRequest.Prompt);
                return new OkObjectResult(result);
            }
            case "PluginTest":
            {
                var (_, result) = await _aiPluginTestService.Ask(promptRequest.Prompt);
                return new OkObjectResult(result);
            }
            case "Rf1350":
            {
                var (_, result) = await _aiArntIvanService.Ask<Rf1350ChatResult>(promptRequest.Prompt);
                return new OkObjectResult(result);
            }
            default:
                return new BadRequestObjectResult("Unknown agent");
        }
    }

    private async Task BlobStorageFailedCount()
    {
        var failedBlobs = await _blobService.ListBlobs(_blobStorageFailedName);
        _metricsService.Gauge($"{Constants.MetricsPrefix}_BlobStorageCount", "Current count of blobs in storage", failedBlobs.Count, ("BlobStorage", _blobStorageFailedName));
        _logger.LogDebug("BlobStorage {BlobStorageFailedName} has {Count} blobs", _blobStorageFailedName, failedBlobs.Count);
    }
    
    private async Task<List<BlobItem>> BlobStorageQueueCount()
    {
        var mailBlobs = await _blobService.ListBlobs(_blobStorageQueueName);
        _metricsService.Gauge($"{Constants.MetricsPrefix}_BlobStorageCount", "Current count of blobs in storage", mailBlobs.Count, ("BlobStorage", _blobStorageQueueName));
        _logger.LogDebug("BlobStorage {BlobStorageQueueName} has {Count} blobs", _blobStorageQueueName, mailBlobs.Count);
        
        return mailBlobs;
    }

    private async Task MovedByRulesCount()
    {
        // count unwanted emails moved by rule
        const string unwantedLastRegisteredDateTime = "unwantedLastRegisteredDateTime";
        var lastRegisteredReceivedDateTime =
            await _mongoDbServices.GetLastRegisteredDateTime(unwantedLastRegisteredDateTime);
        var unwantedMailMessages =
            await GetFolderEmails(_postboxUpn, _mailFolderUnwantedRuleId, lastRegisteredReceivedDateTime);
        if (unwantedMailMessages.Count > 0)
        {
            lastRegisteredReceivedDateTime = unwantedMailMessages.Max(message => message.ReceivedDateTime ?? lastRegisteredReceivedDateTime);
            _metricsService.Count($"{Constants.MetricsPrefix}_UnwantedMailMessageCount",
                "Count of unwanted mail messages moved by rule", unwantedMailMessages.Count);
            
            await _mongoDbServices.UpdateLastRegisteredDateTime(unwantedLastRegisteredDateTime, lastRegisteredReceivedDateTime);
            await _statisticsService.InsertRuleStatistics("Unwanted mail messages moved by rule", "UnwantedMailMessagesRule",
                unwantedMailMessages.Count);
        }
    }
    
    private async Task<List<Message>> GetFolderEmails(string userPrincipalName, string folderId, DateTimeOffset lastRegisteredReceivedDateTime)
    {
        /*
         * NOTE: Exchange Server stores receivedDateTime with milliseconds precision
         * and MS Graph does not return milliseconds in the returned dateTime properties, so we must filter it ourselves.
         */
        var mailMessages = await _graphService.GetMailMessages(userPrincipalName, folderId,
            filter: $"receivedDateTime gt {HelperTools.GetUtcDateTimeString(lastRegisteredReceivedDateTime)}", top: 999);

        var filteredMailMessages = mailMessages
            .Where(message => message.ReceivedDateTime.HasValue &&
                              message.ReceivedDateTime.Value > lastRegisteredReceivedDateTime)
            .ToList();
        
        _logger.LogInformation("Filtered folder mail messages to {Count} messages", filteredMailMessages.Count);

        return filteredMailMessages;
    }
    
    private async Task<ArchiveOkResponse> GetAndHandleEmails()
    {
        _metricsService.Count($"{Constants.MetricsPrefix}_GetAndHandleEmails", "Count of get and handle emails started");
        
        var mailBlobs = await BlobStorageQueueCount();

        var mailMessages = await _graphService.GetMailMessages(_postboxUpn, _mailFolderInboxId);
        if (mailMessages.Count == 0)
        {
            _logger.LogInformation("No messages found in Inbox folder");
            return new ArchiveOkResponse();
        }

        List<(IEmailType, FlowStatus)> messagesToHandle = new();
        List<UnknownMessage> unknownMessages = new();

        foreach (var message in mailMessages)
        {
            BlobItem? blobItem = mailBlobs.Find(blob => blob.Name.Contains(message.Id!));

            if (blobItem is not null)
            { 
                var flowStatus = await _blobService.DownloadBlobContent<FlowStatus?>(blobItem.Name);
                if (flowStatus is null)
                {
                     _logger.LogError("Failed to download blob content for BlobName {BlobName}. Blob might be hold on (be strong) to longer than needed. Check blob storage container retention", blobItem.Name);
                     _metricsService.Count($"{Constants.MetricsPrefix}_FlowStatusBlobDownloadFailed", "FlowStatus blob download failed");
                     continue;
                }

                if (flowStatus.SendToArkivarerForHandling)
                {
                    // TODO: Remove with time
                    _metricsService.Count($"{Constants.MetricsPrefix}_SendToArkivarerForHandling", "Send to arkivarer for handling", ("EmailType", flowStatus.Type));
                    _logger.LogError("Hit kommer vi absolutt aldri! MessageId {MessageId} is unhandelable. Send to arkivarer for handling", message.Id);
                    continue;
                }

                if (flowStatus.RetryAfter is not null && flowStatus.RetryAfter > DateTime.UtcNow)
                {
                     continue;
                }
                
                IEmailType existingEmailType = _emailTypeService.CreateEmailTypeInstance(flowStatus.Type);

                messagesToHandle.Add((existingEmailType, flowStatus));
                continue;
            }
            
            try
            {
                var (emailType, unknownMessage) = await _emailTypeService.GetEmailType(message);
                if (unknownMessage is not null)
                {
                    unknownMessages.Add(unknownMessage);
                    continue;
                }

                if (emailType is null)
                {
                    throw new InvalidOperationException(
                        $"No email type found for message with Id {message.Id}. This should never happen, please check your email types.");
                }

                messagesToHandle.Add((emailType, new FlowStatus
                {
                    Type = emailType.GetType().Name,
                    Message = message
                }));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to determine email type for MessageId {MessageId}.", message.Id);
            }
        }
        
        await HandleUnknownMessages(unknownMessages);
        
        await HandleKnownMessageTypes(messagesToHandle);

        var okResponse = new ArchiveOkResponse
        {
            HandledMessages = messagesToHandle.Select(message => new HandledMessage
            {
                MessageId = message.Item2.Message.Id,
                Type = message.Item2.Type
            }).ToList(),
            UnhandledMessageIds = unknownMessages.Select(unknownMessage => unknownMessage.Message.Id).ToList()
        };

        _logger.LogInformation("Handled {HandledMessageCount} messages. {UnknownMessageCount} messages are unhandled",
            messagesToHandle.Count, unknownMessages.Count);

        return okResponse;
    }
    
    [SuppressMessage("ReSharper", "StructuredMessageTemplateProblem")]
    private async Task HandleKnownMessageTypes(List<(IEmailType, FlowStatus)> messagesToHandle)
    {
        foreach ((IEmailType emailType, FlowStatus flowStatus) in messagesToHandle)
        {
            using (GlobalLogContext.PushProperty("MessageId", flowStatus.Message.Id))
            using (GlobalLogContext.PushProperty("EmailType", flowStatus.Type))
            {
                try
                {
                    _logger.LogInformation("Starting {EmailType}.HandleMessage for MessageId {MessageId}");
                    var handledMessage = await emailType.HandleMessage(flowStatus);
                    _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Handled", "EmailType handled", ("EmailType", flowStatus.Type), ("Result", "Success"));
                    
                    var funFact = emailType.IncludeFunFact ? await _aiArntIvanService.FunFact() : string.Empty;
                    var funFactMessage = emailType.IncludeFunFact && !string.IsNullOrEmpty(funFact)
                        ? $"<br /><b>FunFact</b>: {funFact}"
                        : string.Empty;

                    string comment = HelperTools.GenerateHtmlBox($@"
                                    <b>Type e-post</b>: {emailType.Title}<br />
                                    <b>Håndtert</b>: <i>{HelperTools.GetDateTimeOffset():dd.MM.yyyy HH:mm:ss}</i><br />
                                    <b>Info</b>: {handledMessage}<br />
                                    <b>AI-resultat</b>: {emailType.Result}<br />
                                    {funFactMessage}");

                    // update body to reflect that its handled
                    Message message = new Message
                    {
                        Body = new ItemBody
                        {
                            Content = $"{comment}<br /><br />{flowStatus.Message.Body!.Content}",
                            ContentType = BodyType.Html
                        },
                    };

                    await _graphService.PatchMailMessage(_postboxUpn, flowStatus.Message.Id!, message);

                    await _graphService.MoveMailMessage(_postboxUpn, flowStatus.Message.Id!, _mailFolderFinishedId);

                    await RemoveFlowStatusBlob(flowStatus.Type, flowStatus.Message.Id!);
                    
                    await _statisticsService.InsertSystemStatistics(handledMessage, flowStatus.Type, flowStatus.Message.Id!, flowStatus.Message.From?.EmailAddress?.Address);
                    
                    _logger.LogInformation("Finished {EmailType}.HandleMessage for MessageId {MessageId} with result {HandledMessage}",
                        flowStatus.Type, flowStatus.Message.Id, handledMessage);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error handling MessageId {MessageId} with {EmailType}");
                    _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Handled", "EmailType handled", ("EmailType", flowStatus.Type), ("Result", "Failed"));
                    
                    flowStatus.ErrorMessage = ex.Message;
                    flowStatus.ErrorStack = ex.StackTrace;
                    flowStatus.RunCount++;

                    if (flowStatus.SendToArkivarerForHandling || flowStatus.RunCount > _retryMinutesIntervals.Length)
                    {
                        _logger.LogWarning(
                            "MessageId {MessageId} is unhandelable. Message will be moved to manual handling folder and arkivarer must do something!");
                        _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Unhandelable", "EmailType unhandelable", ("EmailType", flowStatus.Type));
                        
                        await _graphService.MoveMailMessage(_postboxUpn, flowStatus.Message.Id!,
                            _mailFolderManualHandlingId);

                        await UpsertFlowStatusBlob(flowStatus.Message.Id!, flowStatus, _blobStorageFailedName);
                        await RemoveFlowStatusBlob(flowStatus.Type, flowStatus.Message.Id!);
                        continue;
                    }

                    int retryAfterMinutes = _retryMinutesIntervals[flowStatus.RunCount - 1];
                    flowStatus.RetryAfter = DateTime.UtcNow.AddMinutes(retryAfterMinutes);
                    _logger.LogWarning(
                        "MessageId {MessageId} will be retried after {RetryAfterMinutes} minutes ({RetryAfter}). RunCount: {RunCount}",
                        flowStatus.Message.Id, retryAfterMinutes, flowStatus.RetryAfter, flowStatus.RunCount);

                    await UpsertFlowStatusBlob(flowStatus.Message.Id!, flowStatus);
                }
            }
        }
    }

    private async Task HandleUnknownMessages(List<UnknownMessage> unknownMessages)
    {
        foreach (var unknownMessage in unknownMessages)
        {
            var destinationId = unknownMessage.PartialMatch
                ? _mailFolderMaybeId
                : _mailFolderNoMatchId;
            
            unknownMessage.Message.Body!.Content = $"{unknownMessage.Result}{unknownMessage.Message.Body!.Content}";
            unknownMessage.Message.Body!.ContentType = BodyType.Html;
            
            _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Unknown", "EmailType unknown", ("PartialMatch", unknownMessage.PartialMatch ? "Yes" : "No"));

            var newMessage =
                await _graphService.CreateMailMessage(_postboxLogUpn, destinationId, unknownMessage.Message);
            if (newMessage is not null)
            {
                _logger.LogInformation(
                    unknownMessage.PartialMatch
                        ? "MessageId {MessageId} is partially matched and a copy is created in {LogUpn} @ PartialMatch folder"
                        : "MessageId {MessageId} is not matched and a copy is created in {LogUpn} @ NoMatch folder",
                    unknownMessage.Message.Id, _postboxLogUpn);
            }
            
            _logger.LogInformation("MessageId {MessageId} is of an unknown type and will be moved to manual handling folder", unknownMessage.Message.Id);
            
            await _statisticsService.InsertSystemStatistics("Email with unknown type", "Unknown", unknownMessage.Message.Id!, unknownMessage.Message.From?.EmailAddress?.Address);
            
            await _graphService.MoveMailMessage(_postboxUpn, unknownMessage.Message.Id!, _mailFolderManualHandlingId);
        }
    }
    
    private async Task UpsertFlowStatusBlob(string messageId, FlowStatus flowStatus, string? folder = null)
    {
        try
        {
            folder ??= _blobStorageQueueName;
            await _blobService.UploadBlob($"{folder}/{flowStatus.Type}/{messageId}-flowstatus.json",
                JsonSerializer.Serialize(flowStatus, new JsonSerializerOptions
                {
                    IndentSize = 2,
                    WriteIndented = true
                }));
            
            _metricsService.Count($"{Constants.MetricsPrefix}_FlowStatusBlob_Upserted", "FlowStatus blob upserted", ("EmailType", flowStatus.Type), ("Result", "Success"));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to serialize flowStatus or upload blob for MessageId {MessageId}. Things might happen multiple times", messageId);
            _metricsService.Count($"{Constants.MetricsPrefix}_FlowStatusBlob_Upserted", "FlowStatus blob upserted", ("EmailType", flowStatus.Type), ("Result", "Failed"));
        }
    }
    
    private async Task RemoveFlowStatusBlob(string flowType, string messageId, string? folder = null)
    {
        folder ??= _blobStorageQueueName;
        var blobPath = $"{folder}/{flowType}/{messageId}-flowstatus.json";
        
        try
        {
            var blobCount = await _blobService.RemoveBlobs(blobPath);
            
            if (blobCount > 0)
            {
                _metricsService.Count($"{Constants.MetricsPrefix}_FlowStatusBlob_Removed", "FlowStatus blob removed", blobCount, ("EmailType", flowType), ("Result", "Success"));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to remove blob with BlobPath {BlobPath}. Blob might be hold on (be strong) to longer than needed. Check blob storage container retention", blobPath);
            _metricsService.Count($"{Constants.MetricsPrefix}_FlowStatusBlob_Removed", "FlowStatus blob removed", ("EmailType", flowType), ("Result", "Failed"));
        }
    }
}