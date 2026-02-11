using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Graph.Models;
using postmottak_arkivering_dotnet.Contracts;
using postmottak_arkivering_dotnet.Contracts.Ai.ChatResult;
using postmottak_arkivering_dotnet.Contracts.Email;
using postmottak_arkivering_dotnet.Services;
using postmottak_arkivering_dotnet.Services.Ai;
using Vestfold.Extensions.Archive.Services;
using Vestfold.Extensions.Metrics.Services;

namespace postmottak_arkivering_dotnet.EmailTypes;

public partial class Rf1350EmailType : IEmailType
{
    private readonly IAiArntIvanService _aiArntIvanService;
    private readonly IArchiveService _archiveService;
    private readonly IGraphService _graphService;
    private readonly ILogger<Rf1350EmailType> _logger;
    private readonly IMetricsService _metricsService;
    
    private const string FromAddress = "ikkesvar@regionalforvaltning.no";

    private readonly string[] _subjects = [
        "RF13.50 - Automatisk kvittering på innsendt søknad",
        "RF13.50 - Automatisk epost til arkiv"
    ];
    
    private readonly List<string> _caseStatuses = [
        "Under behandling",
        "Reservert"
    ];

    private readonly string _epostInnDocumentCategory = "";
    private readonly string _postmottakUpn = "";
    private readonly string? _testProjectNumber = "";
    
    private const string AnmodningOmUtbetaling = "Anmodning om utbetaling";
    private const string AutomatiskKvitteringPaInnsendtSoknad = "Automatisk kvittering på innsendt søknad";
    private const string OverforingAvMottattSoknad = "Overføring av mottatt søknad";

    private Rf1350ChatResult? _result;

    public bool Enabled => true;
    public bool IncludeFunFact => false;
    public string Result => JsonSerializer.Serialize(_result, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
    public string Title => "RF13.50";

    public Rf1350EmailType(IServiceProvider serviceProvider)
    {
        _logger = serviceProvider.GetService<ILogger<Rf1350EmailType>>()!;
        _aiArntIvanService = serviceProvider.GetService<IAiArntIvanService>()!;
        _archiveService = serviceProvider.GetService<IArchiveService>()!;
        _graphService = serviceProvider.GetService<IGraphService>()!;
        _metricsService = serviceProvider.GetService<IMetricsService>()!;
        
        if (!Enabled)
        {
            return;
        }

        var configuration = serviceProvider.GetRequiredService<IConfiguration>();
        _epostInnDocumentCategory = configuration["ARCHIVE_DOCUMENT_CATEGORY_EPOST_INN"] ?? throw new NullReferenceException();
        _postmottakUpn = configuration["POSTMOTTAK_UPN"] ?? throw new NullReferenceException();
        _testProjectNumber = configuration["EMAILTYPE_RF1350_TEST_PROJECTNUMBER"];
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
                Result = $"Avsender er ikke {FromAddress}. Dette er ikke en {Title} e-post"
            };
        }

        if (!_subjects.Any(subject => message.Subject!.StartsWith(subject, StringComparison.OrdinalIgnoreCase)))
        {
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.No,
                Result = $"E-postens emne inneholder ikke et gyldig søkeord for {Title.ToLower()}. Gyldige søkeord er: {string.Join(", ", _subjects)}"
            };
        }
        var (_, result) = await _aiArntIvanService.Ask<Rf1350ChatResult>(message.Body!.Content!);
        if (string.IsNullOrEmpty(result?.Type) || string.IsNullOrEmpty(result.ReferenceNumber))
        {
            _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Maybe_Match", "EmailType hit a maybe match", ("EmailType", nameof(Rf1350EmailType)));
            var resultString = JsonSerializer.Serialize(result);
            return new EmailTypeMatchResult
            {
                Matched = EmailTypeMatched.Maybe,
                Result = $"Avsender og emne samsvarte med {Title}, men AI-resultatet indikerer at det ikke er en {nameof(Rf1350EmailType)}:<br />AI-resultat:<br />{resultString}"
            };
        }

        _result = result;
        if (!string.IsNullOrEmpty(_testProjectNumber) && !string.IsNullOrEmpty(_result.ProjectNumber))
        {
            _logger.LogWarning("Test project number {_testProjectNumber} is set. Project number found by Arnt Ivan will be overridden for test purposes", _testProjectNumber);
            _result.ProjectNumber = _testProjectNumber;
        }
        _metricsService.Count($"{Constants.MetricsPrefix}_EmailType_Match", "EmailType hit a maybe match", ("EmailType", nameof(Rf1350EmailType)));
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
            _result = JsonSerializer.Deserialize<Rf1350ChatResult>(flowStatus.Result.ToString()!);
        }
        
        if (_result is null)
        {
            throw new InvalidOperationException("Result is null. Somethings wrong");
        }

        if (_result.Type.Equals(OverforingAvMottattSoknad, StringComparison.OrdinalIgnoreCase))
        {
            return await HandleOverforingAvMottattSoknad(flowStatus);
        }
        
        if (_result.Type.Equals(AutomatiskKvitteringPaInnsendtSoknad, StringComparison.OrdinalIgnoreCase))
        {
            return await HandleAutomatiskKvitteringPaInnsendtSoknad(flowStatus);
        }
        
        if (_result.Type.Equals(AnmodningOmUtbetaling, StringComparison.OrdinalIgnoreCase))
        {
            return await HandleAnmodningOmSluttutbetaling(flowStatus);
        }
        
        throw new InvalidOperationException($"Unknown {Title} type {_result.Type}");
    }
    
    private async Task<string> HandleOverforingAvMottattSoknad(FlowStatus flowStatus)
    {   
        if(string.IsNullOrEmpty(_result!.ProjectNumber))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"Project number {_result.ProjectNumber} is empty or null");
        }
        if (!RegexProjectNumber().IsMatch(_result.ProjectNumber))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"Project number {_result.ProjectNumber} is not valid");
        }
        
        if (!string.IsNullOrEmpty(_result.ReferenceNumber) && !RegexReferenceNumber().IsMatch(_result.ReferenceNumber))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"Reference number {_result.ReferenceNumber} is not valid");
        }
        
        if (string.IsNullOrEmpty(_result!.ProjectOwner))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new MissingFieldException("Project owner is missing");
        }
        
        if (!_result!.ValidOrganizationNumber())
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException("Organization number is missing or invalid");
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
            var projects = await _archiveService.GetProjects(new
            {
                _result!.ProjectNumber
            });

            var activeProject = projects.FirstOrDefault();
            
            if (activeProject is null)
            {
                throw new InvalidOperationException($"No projects found for the given project number {_result.ProjectNumber}");
            }
            
            flowStatus.Archive.Project = activeProject;
            
            var cases = await _archiveService.GetCases(new
            {
                _result!.ProjectNumber,
                Title = $"RF13.50%{_result!.ReferenceNumber}%"
            });

            var activeCase = cases.FirstOrDefault(c => c is not null && _caseStatuses.Contains(c["Status"]!.ToString())) ?? await CreateCase(flowStatus, activeProject);

            flowStatus.Archive.CaseNumber = activeCase["CaseNumber"]!.ToString();
        }
        
        if (string.IsNullOrEmpty(flowStatus.Archive.DocumentNumber))
        {
            await CreateDocument(flowStatus, "Søknad");
        }

        string caseHandle = flowStatus.Archive.CaseCreated
            ? "Sak ble også automatisk opprettet siden robåten ikke fant en eksisterende sak."
            : "Robåten fant en eksisterende sak og arkiverte dokumentet i denne.";
        return $"Overføring av mottatt søknad er automatisk arkivert med dokumentnummer {flowStatus.Archive.DocumentNumber}. {caseHandle}";
    }
    
    private async Task<string> HandleAutomatiskKvitteringPaInnsendtSoknad(FlowStatus flowStatus)
    {
        if (string.IsNullOrEmpty(flowStatus.Archive.CaseNumber) || flowStatus.Archive.SoknadSender is null)
        {
            var cases = await _archiveService.GetCases(new
            {
                Title = $"RF13.50%{_result!.ReferenceNumber}%"
            });

            var activeCase = cases.FirstOrDefault(c => c is not null && _caseStatuses.Contains(c["Status"]!.ToString()));

            if (activeCase?["Documents"] is null)
            {
                throw new InvalidOperationException($"No cases or documents found for the given reference number {_result.ReferenceNumber}. Wait for it to be created.");
            }
            
            flowStatus.Archive.Case = activeCase;
            flowStatus.Archive.CaseNumber = activeCase["CaseNumber"]!.ToString();

            var documentTitle = "RF13.50 - Søknad -";
            var epostInnDocumentCategory = _epostInnDocumentCategory.Replace("recno:", "");
            var caseDocuments = activeCase["Documents"]!.AsArray();
            var caseDocument = caseDocuments.FirstOrDefault(d => d!["DocumentTitle"]!.ToString().StartsWith(documentTitle, StringComparison.OrdinalIgnoreCase) && d["Category"]!["Recno"]!.ToString() == epostInnDocumentCategory);
            if (caseDocument is null)
            {
                throw new InvalidOperationException($"No document that starts with title {documentTitle} found on caseNumber {flowStatus.Archive.CaseNumber}");
            }
            
            var documents = await _archiveService.GetDocuments(new
            {
                DocumentNumber = caseDocument["DocumentNumber"]!.ToString()
            });
            
            var soknadDocument = documents.FirstOrDefault();
            if (soknadDocument?["Contacts"] is null)
            {
                flowStatus.SendToArkivarerForHandling = true;
                throw new InvalidOperationException($"No lame document or Contacts found for the given document number {caseDocument["DocumentNumber"]}");
            }

            flowStatus.Archive.SoknadSender = soknadDocument["Contacts"]!.AsArray().FirstOrDefault(c => c!["Role"]!.ToString() == "Avsender");
            if (flowStatus.Archive.SoknadSender?["ReferenceNumber"] is null)
            {
                flowStatus.SendToArkivarerForHandling = true;
                throw new InvalidOperationException($"No sender or ReferenceNumber found for the given document number {caseDocument["DocumentNumber"]}");
            }
        }
        
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
                Category = "E-post ut",
                Contacts = new object[]
                {
                    new
                    {
                        ReferenceNumber = flowStatus.Archive.SoknadSender!["ReferenceNumber"]!.ToString(),
                        Role = "Mottaker"
                    }
                },
                DocumentDate = flowStatus.Message.ReceivedDateTime.HasValue
                    ? flowStatus.Message.ReceivedDateTime.Value.ToString("O")
                    : DateTime.Now.ToString("O"),
                Files = new List<object>
                {
                    new
                    {
                        Format = "eml",
                        Status = "F",
                        Title = flowStatus.Message.Subject!,
                        Data = messageBytes,
                        VersionFormat = "P"
                    }
                },
                ResponsiblePersonEmail = flowStatus.Archive.Case!["ResponsiblePerson"]!["Email"]!.ToString(),
                Status = "J",
                Title = "RF13.50 - Kvittering på søknad"
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
        }
        
        return $"Kvittering på søknad er automatisk arkivert med dokumentnummer {flowStatus.Archive.DocumentNumber}.";
    }
    
    private async Task<string> HandleAnmodningOmSluttutbetaling(FlowStatus flowStatus)
    {
        if (!string.IsNullOrEmpty(_result!.ProjectNumber) && !RegexProjectNumber().IsMatch(_result.ProjectNumber))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"Project number {_result.ProjectNumber} is not valid");
        }
        
        if (!string.IsNullOrEmpty(_result.ReferenceNumber) && !RegexReferenceNumber().IsMatch(_result.ReferenceNumber))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"Reference number {_result.ReferenceNumber} is not valid");
        }
        
        if (string.IsNullOrEmpty(_result!.ProjectOwner))
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new MissingFieldException("Project owner is missing");
        }
        
        if (!_result!.ValidOrganizationNumber())
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException("Organization number is missing or invalid");
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
            var projects = await _archiveService.GetProjects(new
            {
                _result!.ProjectNumber
            });

            var activeProject = projects.FirstOrDefault();
            
            if (activeProject is null)
            {
                throw new InvalidOperationException($"No projects found for the given project number {_result.ProjectNumber}");
            }
            
            flowStatus.Archive.Project = activeProject;
            
            var cases = await _archiveService.GetCases(new
            {
                _result!.ProjectNumber,
                Title = $"RF13.50%{_result!.ReferenceNumber}%"
            });

            var activeCase = cases.FirstOrDefault(c => c is not null && _caseStatuses.Contains(c["Status"]!.ToString())) ?? await CreateCase(flowStatus, activeProject);

            flowStatus.Archive.CaseNumber = activeCase["CaseNumber"]!.ToString();
        }
        
        if (string.IsNullOrEmpty(flowStatus.Archive.DocumentNumber))
        {
            await CreateDocument(flowStatus, "Anmodning om utbetaling");
        }

        var caseHandle = flowStatus.Archive.CaseCreated
            ? "Sak ble også automatisk opprettet siden robåten ikke fant en eksisterende sak."
            : "Robåten fant en eksisterende sak og arkiverte dokumentet i denne.";
        return $"Anmodning om utbetaling er automatisk arkivert med dokumentnummer {flowStatus.Archive.DocumentNumber}. {caseHandle}";
    }
    
    private async Task<JsonNode> CreateCase(FlowStatus flowStatus, JsonNode activeProject)
    {
        if (int.TryParse(_result!.ReferenceNumber.Split('-')[0].Trim(), out var year) && year < 2024)
        {
            flowStatus.SendToArkivarerForHandling = true;
            throw new InvalidOperationException($"The year {year} probably indicates that this is a case from before 2024 and can be Telemark or Vestfold. Who knows? 🤷‍♂️ Message must be handled manually by to arkivarer.");
        }
        
        var responsiblePersonEmail = activeProject["ResponsiblePerson"]!["Email"]!.ToString();
        if (string.IsNullOrEmpty(responsiblePersonEmail))
        {
            throw new MissingFieldException($"Responsible person email is missing from ProjectNumber {_result!.ProjectNumber}");
        }
                
        var activeCase = await _archiveService.CreateCase(new
        {
            ArchiveCodes = new object[]
            {
                new {
                    ArchiveCode = "243",
                    ArchiveType = "FELLESKLASSE PRINSIPP",
                    Sort = 1
                },
                new {
                    ArchiveCode = "U01",
                    ArchiveType = "FAGKLASSE PRINSIPP",
                    Sort = 2
                }
            },
            Project = _result!.ProjectNumber,
            ResponsiblePersonEmail = responsiblePersonEmail,
            Status = "B",
            Title = $"RF13.50 - {_result.ProjectName} - {_result!.ReferenceNumber} - {_result!.ProjectOwner}"
        });

        flowStatus.Archive.CaseCreated = true;
        
        _metricsService.Count($"{Constants.MetricsPrefix}_CreateCase", "Archive case created", ("EmailType", nameof(Rf1350EmailType)));
        
        return activeCase;
    }
    
    private async Task CreateDocument(FlowStatus flowStatus, string title)
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
                    ReferenceNumber = flowStatus.Archive.SyncEnterprise!["EnterpriseNumber"]!.ToString(),
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
                    Format = "eml",
                    Status = "F",
                    Title = flowStatus.Message.Subject!,
                    Data = messageBytes,
                    VersionFormat = "P"
                }
            },
            ResponsiblePersonEmail = flowStatus.Archive.Project!["ResponsiblePerson"]!["Email"]!.ToString(),
            Status = "J",
            Title = $"RF13.50 - {title} - {_result!.ProjectName} - {_result.ReferenceNumber} - {_result.ProjectOwner}",
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
        
        _metricsService.Count($"{Constants.MetricsPrefix}_CreateDocument", "Archive document created", ("EmailType", nameof(Rf1350EmailType)));

        flowStatus.Archive.DocumentNumber = document["DocumentNumber"]!.ToString();
    }

    [GeneratedRegex(@"^(\d{2})-(\d{1,6})$")]
    private static partial Regex RegexProjectNumber();
    
    [GeneratedRegex(@"^(\d{4})-(\d{4})$")]
    private static partial Regex RegexReferenceNumber();
}