// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text.Encodings.Web;
// using System.Text.Json;
// using System.Threading.Tasks;
// using Microsoft.Extensions.Configuration;
// using Microsoft.Extensions.DependencyInjection;
// using Microsoft.Graph.Models;
// using postmottak_arkivering_dotnet.Contracts;
// using postmottak_arkivering_dotnet.Contracts.Ai.ChatResult;
// using postmottak_arkivering_dotnet.Contracts.Email;
// using postmottak_arkivering_dotnet.Services;
// using postmottak_arkivering_dotnet.Services.Ai;
// using postmottak_arkivering_dotnet.Utils;
// using Vestfold.Extensions.Metrics.Services;

// namespace postmottak_arkivering_dotnet.EmailTypes;

// public class InnsynEmailType : IEmailType
// {
//     private readonly IAiArntIvanService _aiArntIvanService;
//     private readonly IGraphService _graphService;
//     private readonly IMetricsService _metricsService;

//     private readonly string _postmottakUpn = "";
//     private readonly string[] _subjects = [
//         "Innsyn"
//     ];

//     private readonly List<string> _toRecipients = [];

//     private InnsynChatResult? _result;
    
//     public bool Enabled => false;
//     public bool IncludeFunFact => false;
//     public string Result => JsonSerializer.Serialize(_result, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
//     public string Title => "Innsyn";

//     public InnsynEmailType(IServiceProvider serviceProvider)
//     {
//         _aiArntIvanService = serviceProvider.GetService<IAiArntIvanService>()!;
//         _graphService = serviceProvider.GetService<IGraphService>()!;
//         _metricsService = serviceProvider.GetService<IMetricsService>()!;
        
//         if (!Enabled)
//         {
//             return;
//         }
        
//         var configuration = serviceProvider.GetService<IConfiguration>()!;
//         _postmottakUpn = configuration["POSTMOTTAK_UPN"] ?? throw new NullReferenceException();
//         _toRecipients = configuration["EMAILTYPE_INNSYN_ADDRESSES"]?.Split(',').ToList() ?? throw new NullReferenceException();
//     }
    
//     public async Task<EmailTypeMatchResult> MatchCriteria(Message message)
//     {
//         await Task.CompletedTask;

//         if (!_subjects.Any(subject => message.Subject!.Contains(subject, StringComparison.OrdinalIgnoreCase)))
//         {
//             return new EmailTypeMatchResult
//             {
//                 Matched = EmailTypeMatched.No,
//                 Result = "Emnet samsvarer ikke med noen av de forventede emnene"
//             };
//         }
        
//         var (_, result) = await _aiArntIvanService.Ask<InnsynChatResult>(message.Body!.Content!);
//         if (result is null || !result.IsInnsyn)
//         {
//             var resultString = JsonSerializer.Serialize(result);
//             _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Maybe_Match", "EmailType hit a maybe match", ("EmailType", nameof(InnsynEmailType)));
//             return new EmailTypeMatchResult
//             {
//                 Matched = EmailTypeMatched.Maybe,
//                 Result = $"Emne samsvarte med en av de forventede {Title.ToLower()} emnene, men AI-resultatet indikerer at det ikke er en {nameof(InnsynEmailType)}:<br />AI-resultat:<br />{resultString}"
//             };
//         }

//         _result = result;
        
//         _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Match", "EmailType hit a maybe match", ("EmailType", nameof(InnsynEmailType)));
//         return new EmailTypeMatchResult
//         {
//             Matched = EmailTypeMatched.Yes
//         };
//     }

//     public async Task<string> HandleMessage(FlowStatus flowStatus)
//     {
//         if (flowStatus.Result is null)
//         {
//             flowStatus.Result = _result;
//         }
//         else
//         {
//             _result = JsonSerializer.Deserialize<InnsynChatResult>(flowStatus.Result.ToString()!);
//         }
        
//         if (_result is null)
//         {
//             throw new InvalidOperationException("Result is null. Somethings wrong");
//         }
        
//         string forwardMessage = @$"Denne e-posten er håndtert av KI og videresendt på begrunnelse: {_result.Description}.
//                                     <br />Ta kontakt med arkivet dersom du mener at dette er feil.";
        
//         await _graphService.ForwardMailMessage(_postmottakUpn, flowStatus.Message.Id!, _toRecipients, HelperTools.GenerateHtmlBox(forwardMessage));
        
//         _metricsService.Count($"{Constants.MetricsPrefix}_Email_Forwarded", "Email forwarded", ("EmailType", nameof(InnsynEmailType)));
        
//         return $"Denne e-posten er håndtert av KI på begrunnelse: {_result.Description}, og videresendt til <ul>{string.Join("", _toRecipients.Select(recipient => $"<li>{recipient}</li>"))}</ul>";
//     }
// }