using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Graph.Models;
using postmottak_arkivering_dotnet.Contracts;
using postmottak_arkivering_dotnet.Contracts.Ai.ChatResult;
using postmottak_arkivering_dotnet.Contracts.Ai.Enums;
using postmottak_arkivering_dotnet.Contracts.Email;
using postmottak_arkivering_dotnet.Services;
using postmottak_arkivering_dotnet.Services.Ai;
using Vestfold.Extensions.Archive.Services;
using Vestfold.Extensions.Metrics.Services;

namespace postmottak_arkivering_dotnet.EmailTypes;

public class LoyvegarantiEmailType : IEmailType
{
    private readonly IAiArntIvanService _aiArntIvan;
    private readonly IArchiveService _archiveService;
    private readonly IGraphService _graphService;
    private readonly ILogger<LoyvegarantiEmailType> _logger;
    private readonly IMetricsService _metricsService;
    
    private const string FromAddress = "post@matrixinsurance.no";
    
    private readonly List<string> _caseStatuses = [
        "Under behandling",
        "Reservert",
        "Avsluttet"
    ];
    
    private readonly string[] _subjects = [
        "Løyve",
        "Org.nr"
    ];

    private readonly string[] _blackListedSubjects = [
        "Fwd:",
        "FW:",
        "Forward:",
        "Videresend:"
    ];

    private const string MatrixInsuranceReferenceNumber = "966431695";

    private readonly string _epostInnDocumentCategory = "";
    private readonly string _postmottakUpn = "";
    private readonly string _responsibleEnterpriseRecno = "";

    private LoyvegarantiChatResult? _result;

    public bool Enabled => true;
    public bool IncludeFunFact => false;
    public string Result => JsonSerializer.Serialize(_result, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
    public string Title => "Løyvegaranti";

    public LoyvegarantiEmailType(IServiceProvider serviceProvider)
    {
        _aiArntIvan = serviceProvider.GetService<IAiArntIvanService>()!;
        _archiveService = serviceProvider.GetRequiredService<IArchiveService>();
        _graphService = serviceProvider.GetRequiredService<IGraphService>();
        _logger = serviceProvider.GetRequiredService<ILogger<LoyvegarantiEmailType>>();
        _metricsService = serviceProvider.GetRequiredService<IMetricsService>();
        
        if (!Enabled)
        {
            return;
        }
        
        var configuration = serviceProvider.GetRequiredService<IConfiguration>();
        _epostInnDocumentCategory = configuration["ARCHIVE_DOCUMENT_CATEGORY_EPOST_INN"] ?? throw new NullReferenceException();
        _postmottakUpn = configuration["POSTMOTTAK_UPN"] ?? throw new NullReferenceException();
        _responsibleEnterpriseRecno = configuration["EMAILTYPE_LOYVEGARANTI_RESPONSIBLE_ENTERPRISE_RECNO"] ?? throw new NullReferenceException();
    }
    
    public async Task<EmailTypeMatchResult> MatchCriteria(Message message)
    {
        await Task.CompletedTask;
        
        if (string.IsNullOrEmpty(message.From?.EmailAddress?.Address))
        {
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.No,
                Result = "Avsender mangler. WHHAAAT?"
            };
        }
        
        if (!message.From.EmailAddress.Address.Equals(FromAddress, StringComparison.OrdinalIgnoreCase))
        {
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.No,
                Result = $"Avsender er ikke {FromAddress}. Dette er ikke en {Title.ToLower()} e-post"
            };
        }
        
        if (!_subjects.Any(subject => message.Subject!.Contains(subject, StringComparison.OrdinalIgnoreCase)))
        {
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.No,
                Result = $"E-postens emne inneholder ikke et gyldig søkeord for {Title.ToLower()}. Gyldige søkeord er: {string.Join(", ", _subjects)}"
            };
        }
        
        if (_blackListedSubjects.Any(subject => message.Subject!.StartsWith(subject, StringComparison.OrdinalIgnoreCase)))
        {
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.No,
                Result = $"E-postens emne inneholder en ugyldig prefix for {Title.ToLower()}. Ugyldige prefixer er: {string.Join(", ", _blackListedSubjects)}"
            };
        }
        
        var (_, result) = await _aiArntIvan.Ask<LoyvegarantiChatResult>(message.Subject!);
        if (result is null || string.IsNullOrEmpty(result.OrganizationName) || string.IsNullOrEmpty(result.OrganizationNumber) || result.OrganizationNumber.Length != 9 || !int.TryParse(result.OrganizationNumber, out _))
        {
            _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Maybe_Match", "EmailType hit a maybe match", ("EmailType", nameof(LoyvegarantiEmailType)));
            var resultString = JsonSerializer.Serialize(result);
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.Maybe,
                Result = $"Avsender og emne samsvarte med {Title.ToLower()}, men AI-resultatet indikerer at det ikke er en {nameof(LoyvegarantiEmailType)}:<br />AI-resultat:<br />{resultString}"
            };
        }

        _result = result;
        
        _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Match", "EmailType hit a match", ("EmailType", nameof(LoyvegarantiEmailType)));
        return new EmailTypeMatchResult
        {
            Matched = EmailTypeMatched.Yes
        };
    }

    public async Task<string> HandleMessage(FlowStatus flowStatus)
    {
        if (flowStatus.Result is null)
        {
            flowStatus.Result = _result;
        }
        else
        {
            _result = JsonSerializer.Deserialize<LoyvegarantiChatResult>(flowStatus.Result.ToString()!);
        }
        
        if (_result is null)
        {
            throw new InvalidOperationException("Result is null. Somethings wrong");
        }

        if (flowStatus.Archive.SyncEnterprise is null)
        {
            try
            {
                flowStatus.Archive.SyncEnterprise = (await _archiveService.SyncEnterprise(_result.OrganizationNumber.ToString()))["enterprise"];
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("404"))
            {
                flowStatus.SendToArkivarerForHandling = true;
                throw;
            }
        }
        
        if (string.IsNullOrEmpty(flowStatus.Archive.CaseNumber))
        {
            var cases = await _archiveService.GetCases(new
            {
                ArchiveCode = _result.OrganizationNumber,
                Title = $"Drosjeløyve - % - {_result.OrganizationNumber}%"
            });

            var activeCase = cases.FirstOrDefault(c => c is not null && _caseStatuses.Contains(c["Status"]!.ToString()));

            if (activeCase is not null && activeCase["Status"]?.ToString() == "Avsluttet")
            {
                try
                {
                    await _archiveService.UpdateCase(new
                    {
                        CaseNumber = activeCase["CaseNumber"]!.ToString(),
                        Status = "B"
                    });
                    
                    _metricsService.Count($"{Constants.MetricsPrefix}_UpdateCase", "Update case called", ("Result", "Success"));
                    _logger.LogInformation("Updated case status to 'Under behandling' (B) for CaseNumber {CaseNumber}", activeCase["CaseNumber"]);
                }
                catch (InvalidOperationException ex)
                {
                    _metricsService.Count($"{Constants.MetricsPrefix}_UpdateCase", "Update case called", ("Result", "Failed"));
                    _logger.LogError(ex, "Failed to update case status to 'Under behandling' (B) for CaseNumber {CaseNumber}", activeCase["CaseNumber"]);
                    throw;
                }
            }
            
            if (activeCase is null)
            {
                var caseTitle = $"Drosjeløyve - {_result.OrganizationName} - {_result.OrganizationNumber}";
                
                activeCase = await _archiveService.CreateCase(new
                {
                    AccessCode = "U",
                    AccessGroup = "Alle",
                    ArchiveCodes = new object[]
                    {
                        new {
                            ArchiveCode = _result.OrganizationNumber,
                            ArchiveType = "ORG",
                            IsManualText = true,
                            Sort = 1
                        },
                        new {
                            ArchiveCode = "N12",
                            ArchiveType = "FAGKLASSE PRINSIPP",
                            Sort = 2
                        },
                        new {
                            ArchiveCode = "&18",
                            ArchiveType = "TILLEGGSKODE PRINSIPP",
                            Sort = 3
                        }
                    },
                    CaseType = "Sak",
                    ResponsibleEnterpriseRecno = _responsibleEnterpriseRecno,
                    Status = "B",
                    SubArchive = "Løyve",
                    Title = caseTitle
                });
                
                flowStatus.Archive.CaseCreated = true;
                
                _metricsService.Count($"{Constants.MetricsPrefix}_CreateCase", "Archive case created");
                _logger.LogInformation("Created new case with CaseNumber {CaseNumber} and Title {Title}", activeCase["CaseNumber"], caseTitle);
            }

            flowStatus.Archive.CaseNumber = activeCase["CaseNumber"]!.ToString();
        }

        var title = GetTitle();
        
        if (string.IsNullOrEmpty(flowStatus.Archive.DocumentNumber))
        {
            var messageBytes = await _graphService.GetMailMessageRaw(_postmottakUpn, flowStatus.Message.Id!);
            
            List<Attachment> mailAttachments = (bool)flowStatus.Message.HasAttachments!
                ? await _graphService.GetMailMessageAttachments(_postmottakUpn, flowStatus.Message.Id!)
                : [];
            
            List<MailAttachment> attachments = mailAttachments
                .Where(a => a.OdataType == "#microsoft.graph.fileAttachment")
                .Select(_graphService.GetMailAttachment)
                .ToList();
            
            var payload = new
            {
                Archive = "Saksdokument",
                flowStatus.Archive.CaseNumber,
                Category = _epostInnDocumentCategory,
                Contacts = new object[]
                {
                    new
                    {
                        ReferenceNumber = MatrixInsuranceReferenceNumber,
                        Role = "Avsender"
                    }
                },
                DocumentDate = flowStatus.Message.ReceivedDateTime.HasValue
                    ? flowStatus.Message.ReceivedDateTime.Value.ToString("O")
                    : DateTime.Now.ToString("O"),
                Files = new List<object>
                {
                    new
                    {
                        Format = "EML",
                        Status = "F",
                        Title = flowStatus.Message.Subject!,
                        Data = messageBytes,
                        VersionFormat = "P"
                    }
                },
                ResponsibleEnterpriseRecno = _responsibleEnterpriseRecno,
                Status = "J",
                Title = $"{title} - {_result.OrganizationName} - {_result.OrganizationNumber}",
            };
            
            attachments.ForEach(a =>
            {
                var (extension, versionFormat) = _archiveService.GetFileExtension(a.Name);
                payload.Files.Add(new
                {
                    Format = extension,
                    Status = "F",
                    Title = a.Name,
                    Data = a.Content,
                    VersionFormat = versionFormat
                });
            });
            
            var document = await _archiveService.CreateDocument(payload);
            flowStatus.Archive.DocumentNumber = document["DocumentNumber"]!.ToString();
            
            _metricsService.Count($"{Constants.MetricsPrefix}_CreateDocument", "Archive document created", ("EmailType", nameof(LoyvegarantiEmailType)));
            _logger.LogInformation("Created document with DocumentNumber {DocumentNumber} for CaseNumber {CaseNumber}",
                flowStatus.Archive.DocumentNumber, flowStatus.Archive.CaseNumber);
        }
        
        var caseHandle = flowStatus.Archive.CaseCreated
            ? "Sak ble også automatisk opprettet siden robåten ikke fant en eksisterende sak."
            : "Robåten fant en eksisterende sak og arkiverte dokumentet i denne.";
        return await Task.FromResult($"{title} er automatisk arkivert med dokumentnummer {flowStatus.Archive.DocumentNumber}. {caseHandle}");
    }

    private string GetTitle()
    {
        if (_result is null)
        {
            throw new InvalidOperationException("LøyvegarantiType not found. Cannot determine title.");
        }
        
        return _result.Type switch
        {
            LøyveGarantiType.Løyvegaranti => "Løyvegaranti",
            LøyveGarantiType.EndringAvLøyvegaranti => "Endring av løyvegaranti",
            LøyveGarantiType.OpphørAvLøyvegaranti => "Opphør av løyvegaranti",
            _ => throw new ArgumentOutOfRangeException(nameof(_result.Type), _result.Type, null)
        };
    }
}