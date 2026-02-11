using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
//using Azure.Core.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using Microsoft.Graph.Models;
using Microsoft.Graph.Models.ODataErrors;
using Microsoft.Graph.Users.Item.MailFolders.Item.Messages;
using Microsoft.Graph.Users.Item.Messages.Item.Copy;
using Microsoft.Graph.Users.Item.Messages.Item.Forward;
using Microsoft.Graph.Users.Item.Messages.Item.Move;
using Microsoft.Graph.Users.Item.Messages.Item.Reply;
using Microsoft.Kiota.Abstractions;
using postmottak_arkivering_dotnet.Contracts.Email;
using Vestfold.Extensions.Authentication.Services;

namespace postmottak_arkivering_dotnet.Services;

public interface IGraphService
{
    Task<Message?> CopyMailMessage(string userPrincipalName, string messageId, string destinationFolderId);
    Task<Message?> CreateMailMessage(string userPrincipalName, string destinationFolderId, Message message);
    Task ForwardMailMessage(string userPrincipalName, string messageId, List<string> recipients, string? comment = null);
    MailAttachment GetMailAttachment(Attachment attachment);
    Task<List<MailFolder>> GetMailFolders(string userPrincipalName);
    Task<List<MailFolder>> GetMailChildFolders(string userPrincipalName, string folderId);
    Task<byte[]> GetMailMessageRaw(string userPrincipalName, string messageId);
    Task<List<Attachment>> GetMailMessageAttachments(string userPrincipalName, string messageId);
    Task<List<Message>> GetMailMessages(string userPrincipalName, string folderId, string[]? expandedProperties = null,
        string? filter = null, string? orderBy = "receivedDateTime asc", int top = 100);
    Task<Message?> MoveMailMessage(string userPrincipalName, string messageId, string destinationFolderId);
    Task<bool> PatchMailMessage(string userPrincipalName, string messageId, Message message);
    Task<bool> ReplyMailMessage(string userPrincipalName, string messageId,
        string fromAddress, List<string> toAddresses, string replyBody, string conversationId, string parentFolderId);
}

public class GraphService : IGraphService
{
    private readonly GraphServiceClient _graphClient;
    private readonly ILogger<GraphService> _logger;
    
    private const string ImmutableIdHeader = "Prefer";
    private const string ImmutableIdHeaderValue = "IdType=\"ImmutableId\"";

    public GraphService(ILogger<GraphService> logger, IAuthenticationService authenticationService)
    {
        _logger = logger;
        _graphClient = authenticationService.CreateGraphClient();
    }

    public async Task<Message?> CopyMailMessage(string userPrincipalName, string messageId, string destinationFolderId)
    {
        try
        {
            var message = await _graphClient.Users[userPrincipalName].Messages[messageId].Copy.PostAsync(
                new CopyPostRequestBody
                {
                    DestinationId = destinationFolderId
                }, configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));

            if (message is null)
            {
                throw new InvalidOperationException("Message not copied");
            }
            
            _logger.LogInformation("MessageId {MessageId} successfully copied to {CopiedMessageId} in FolderId {DestinationFolderId} in {UserPrincipalName}", messageId, message.Id, destinationFolderId, userPrincipalName);
            return message;
        }
        catch (ODataError ex)
        {
            _logger.LogError(ex, "MessageId {MessageId} failed to be copied to FolderId {DestinationFolderId} in {UserPrincipalName}", messageId, destinationFolderId, userPrincipalName);
            return null;
        }
    }

    public async Task<Message?> CreateMailMessage(string userPrincipalName, string destinationFolderId, Message message)
    {
        try
        {
            var createdMessage = await _graphClient.Users[userPrincipalName].MailFolders[destinationFolderId].Messages.PostAsync(message,
                configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));

            if (createdMessage is null)
            {
                throw new InvalidOperationException("Message not created");
            }
            
            _logger.LogInformation("MessageId {MessageId} successfully created in {UserPrincipalName}", createdMessage.Id, userPrincipalName);
            return createdMessage;
        }
        catch (ODataError ex)
        {
            _logger.LogError(ex, "Failed to create message for user {UserPrincipalName}. Message: {@Message}", userPrincipalName, message);
            return null;
        }
    }
    
    public async Task ForwardMailMessage(string userPrincipalName, string messageId, List<string> recipients, string? comment = null)
    {
        var toRecipients = GetRecipients(recipients);
        if (toRecipients is null || toRecipients.Count == 0)
        {
            throw new ArgumentException("Recipients cannot be null or empty", nameof(recipients));
        }

        await _graphClient.Users[userPrincipalName].Messages[messageId].Forward.PostAsync(new ForwardPostRequestBody
        {
            Comment = comment,
            ToRecipients = toRecipients
        }, configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));
    }
    
    public MailAttachment GetMailAttachment(Attachment attachment)
    {
        FileAttachment? fileAttachment = attachment as FileAttachment;
        if (fileAttachment?.ContentBytes is null || fileAttachment.Name is null)
        {
            throw new InvalidOperationException("Attachment is not a file attachment");
        }
        
        return new MailAttachment(fileAttachment.ContentBytes, fileAttachment.Name);
    }

    public async Task<List<MailFolder>> GetMailFolders(string userPrincipalName)
    {
        var mailFolders = await _graphClient.Users[userPrincipalName].MailFolders.GetAsync(
            configuration => 
            {
                configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue);
                configuration.QueryParameters.Top = 20;
            });
        if (mailFolders?.Value is null)
        {
            return new List<MailFolder>();
        }

        return mailFolders.Value;
    }
    
    public async Task<List<MailFolder>> GetMailChildFolders(string userPrincipalName, string folderId)
    {
        //using AzureEventSourceListener listener = AzureEventSourceListener.CreateConsoleLogger();
        var childFolders = await _graphClient.Users[userPrincipalName].MailFolders[folderId].ChildFolders.GetAsync(
            configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));
        if (childFolders?.Value is null)
        {
            return new List<MailFolder>();
        }

        return childFolders.Value;
    }

    public async Task<byte[]> GetMailMessageRaw(string userPrincipalName, string messageId)
    {
        await using var messageStream = await _graphClient.Users[userPrincipalName].Messages[messageId].Content
            .GetAsync(configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));
        if (messageStream is null)
        {
            throw new InvalidOperationException("Message stream is null");
        }

        return ReadAsBytes(messageStream);
    }
    
    public async Task<List<Attachment>> GetMailMessageAttachments(string userPrincipalName, string messageId)
    {
        var attachments = await _graphClient.Users[userPrincipalName].Messages[messageId].Attachments.GetAsync(
            configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));
        if (attachments?.Value is null)
        {
            return new List<Attachment>();
        }

        return attachments.Value;
    }

    public async Task<List<Message>> GetMailMessages(string userPrincipalName, string folderId,
        string[]? expandedProperties = null, string? filter = null, string? orderBy = "receivedDateTime asc",
        int top = 100)
    {
        /*using AzureEventSourceListener listener = AzureEventSourceListener.CreateConsoleLogger();*/

        var mailMessages = await _graphClient.Users[userPrincipalName].MailFolders[folderId].Messages.GetAsync(Options);
        if (mailMessages?.Value is null)
        {
            return new List<Message>();
        }

        _logger.LogInformation("Retrieved {Count} mail messages from {UserPrincipalName} in folder {FolderId} with filter {Filter}", mailMessages.Value.Count, userPrincipalName, folderId, filter);
        return mailMessages.Value;

        void Options(RequestConfiguration<MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters> config)
        {
            config.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue);
            config.QueryParameters.Expand = expandedProperties;
            config.QueryParameters.Filter = filter;
            config.QueryParameters.Orderby = new[] { orderBy ?? "receivedDateTime asc" };
            config.QueryParameters.Top = top;
        }
    }

    public async Task<Message?> MoveMailMessage(string userPrincipalName, string messageId, string destinationFolderId)
    {
        try
        {
            var message = await _graphClient.Users[userPrincipalName].Messages[messageId].Move.PostAsync(new MovePostRequestBody
            {
                DestinationId = destinationFolderId
            }, configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));

            if (message is null)
            {
                throw new InvalidOperationException("Message not moved");
            }

            return message;
        }
        catch (ODataError ex)
        {
            _logger.LogError(ex, "MessageId {MessageId} failed to be moved to FolderId {DestinationFolderId} in {UserPrincipalName}", messageId, destinationFolderId, userPrincipalName);
            return null;
        }
    }
    
    public async Task<bool> PatchMailMessage(string userPrincipalName, string messageId, Message message)
    {
        try
        {
            await _graphClient.Users[userPrincipalName].Messages[messageId].PatchAsync(message,
                configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));

            return true;
        }
        catch (ODataError ex)
        {
            _logger.LogError(ex, "MessageId {MessageId} failed to be patched in {UserPrincipalName}. Message: {@Message}", messageId, userPrincipalName, message);
            throw;
        }
    }
    
    public async Task<bool> ReplyMailMessage(string userPrincipalName, string messageId,
        string fromAddress, List<string> toAddresses, string replyBody, string conversationId, string parentFolderId)
    {
        var replyRequestBody = new ReplyPostRequestBody
        {
            Message = new Message
            {
                From = new Recipient
                {
                    EmailAddress = new EmailAddress
                    {
                        Address = fromAddress
                    }
                },
                ReplyTo = GetRecipients(toAddresses),
                ToRecipients = GetRecipients(toAddresses),
                ConversationId = conversationId,
                ParentFolderId = parentFolderId,
                IsRead = true,
                SentDateTime = DateTimeOffset.Now,
            },
            Comment = replyBody
        };
        
        try
        {
            await _graphClient.Users[userPrincipalName].Messages[messageId].Reply.PostAsync(replyRequestBody,
                configuration => configuration.Headers.Add(ImmutableIdHeader, ImmutableIdHeaderValue));

            return true;
        }
        catch (ODataError ex)
        {
            _logger.LogError(ex, "MessageId {MessageId} failed to be replied to in {UserPrincipalName}. ReplyBody: {@ReplyBody}", messageId, userPrincipalName, replyRequestBody);
            return false;
        }
    }

    private static List<Recipient>? GetRecipients(List<string>? emailAddresses) =>
        emailAddresses?.Select(toAddress => new Recipient
        {
            EmailAddress = new EmailAddress
            {
                Address = toAddress
            }
        }).ToList();

    private static byte[] ReadAsBytes(Stream input)
    {
        using var memoryStream = new MemoryStream();
        input.CopyTo(memoryStream);
        return memoryStream.ToArray();
    }
}