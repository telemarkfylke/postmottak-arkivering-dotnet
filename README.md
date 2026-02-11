# postmottak-arkivering
Løsning som automatisk arkiverer gjenkjennbare e-poster fra en postboks og inn i arkivsystem

## Behov
Automatisering av innkommende e-post til fylkeskommunen. Hovedsaklig automatisk arkivering av kjente e-posttyper.

## Løsningsskisse
![image](https://github.com/user-attachments/assets/540a4bc0-b92a-4c59-b7d3-2f623c73510d)

## Kjente e-posttyper

### RF13.50 (søknad om midler knyttet til statlige tilskuddsordninger)
[Mer info](https://www.regionalforvaltning.no/Dokumentasjon/index.html?soker.htm)

#### Kriterie for gjenkjenning
- Avsender er ikkesvar@regionalforvaltning.no
- Emnefelt må være en av:
    - RF13.50 - Automatisk kvittering på innsendt søknad
    - RF13.50 - Automatisk epost til arkiv
- KI-modell finner all nødvendig data i epost-innhold

#### Flyt
- Overføring av søknad
    - Sjekk at avsender er en registrert virksomhet i arkivet
    - Sjekk at prosjektet som er oppgitt i e-posten eksisterer
    - Finn eksisterende sak, opprett hvis den mangler
        - Dersom saken må opprettes og søknaden gjelder tidligere enn 2024 må den håndteres manuelt
    - Opprett dokumentet/journalposten basert på e-posten og vedlegg
- Kvittering på innsendt søknad
    - Hent saken
    - Hent original søknad fra saken - og hent ut avsender fra original søknad
    - Opprett dokumentet/journalposten basert på e-posten og vedlegg
- Anmodning om del-/slutt-utbetaling
    - Sjekk at avsender er en registrert virksomhet i arkivet
    - Sjekk at prosjektet som er oppgitt i e-posten eksisterer
    - Finn eksisterende sak, opprett hvis den mangler
        - Dersom saken må opprettes og søknaden gjelder tidligere enn 2024 må den håndteres manuelt
    - Opprett dokumentet/journalposten basert på e-posten og vedlegg

### Pengetransporten (ting som har med faktura å gjøre)

#### Kriterie for gjenkjenning
- Emnefelt må inneholde et ord som har noe med betaling eller faktura å gjøre.
- KI-modell må svare "JA" på at dette enten er en faktura / purre av noe slag, eller et spørsmål om betaling til fylkeskommunen

#### Flyt
- Videresend e-posten til faktura-avdelingen

### Løyvegaranti

#### Kriterie for gjenkjenning
- Epost må komme fra matrixinsurance
- Emne må inneholde "Løyve" og "Org. nr"
- Eposten må ikke være videresendt
- KI må finne nødvendige data for arkivering 

#### Flyt
- Sjekker om det finnes en sak
  - Oppdaterer saken til status "Under behandling" om det trengs
  - Lager ny sak om den ikke finner
- Oppretter nytt dokument i saken basert på eposten, og legger til evt vedlegg
- Tittel velges basert på typen epost (Løyvegaranti, Endring av Løyvegaranti, eller Opphør av Løyvegaranti)
- 👍

## Teknisk skisse
![image](https://github.com/user-attachments/assets/94dd7042-7cec-49bc-8da0-aa37f441bfa5)


## Oppsett
- Lag en outlook regel på innboks-mappen som flytter e-post som skal sjekkes for automatisering/automatiseres til {Postmottak_MailFolder_Inbox_Id}
- Opprett en `local.settings.json` fil med følgende innhold og bytt ut med riktige verdier:

> Optional properties:
> - `AppName`: If not set, the assembly name will be used
> - `Version`: If not set, the assembly version will be used
> - `EMAILTYPE_RF13.50_TEST_PROJECTNUMBER`: Only needed in dev and local, and should refer to a test project in the test environment

```json
{
    "IsEncrypted": false,
    "Values": {
        "AzureWebJobsStorage": "UseDevelopmentStorage=true",
        "FUNCTIONS_WORKER_RUNTIME": "dotnet-isolated",
        "SynchronizeSchedule": "0 */5 * * * *",
        "BLOB_STORAGE_CONNECTION_STRING": "connstring",
        "BLOB_STORAGE_CONTAINER_NAME": "dev-local",
        "BLOB_STORAGE_QUEUE_NAME": "queue",
        "BLOB_STORAGE_FAILED_NAME": "failed",
        "AppName": "postmottak-arkivering-dotnet",
        "Version": "0.0.1",
        "BetterStack_Endpoint": "https://something.betterstackdata.com",
        "BetterStack_SourceToken": "super_secret",
        "MicrosoftTeams_WebhookUrl": "microsoft teams webhook url | microsoft power automate flow url if UseWorkflows is set to true",
        "MicrosoftTeams_UseWorkflows": "true if Microsoft Power Automate flow is used, false if Microsoft Teams webhook is used (default is true)",
        "MicrosoftTeams_TitleTemplate": "The title template of the card",
        "MicrosoftTeams_MinimumLevel": "Warning",
        "Serilog_MinimumLevel_Override_Microsoft.Hosting": "Warning",
        "Serilog_MinimumLevel_Override_Microsoft.AspNetCore": "Warning",
        "Serilog_MinimumLevel_Override_OpenApiTriggerFunction": "Warning",
        "RETRY_INTERVALS": "Her legger du en kommaseparert liste med antall minutter mellom hver retry (1,1,1,1)",
        "AZURE_CLIENT_ID": "AppRegClientId",
        "AZURE_CLIENT_SECRET": "AppRegClientSecret",
        "AZURE_TENANT_ID": "AppRegTenantId",
        "ARCHIVE_SCOPE": "arkivskop",
        "ARCHIVE_BASE_URL": "url-til-arkiv-api",
        "ARCHIVE_DOCUMENT_CATEGORY_EPOST_INN": "epost-inn-recno",
        "POSTMOTTAK_UPN": "mailbox@domain.no",
        "POSTMOTTAK_LOG_UPN": "mailbox@domain.no",
        "POSTMOTTAK_MAIL_FOLDER_INBOX_ID": "Inbox folder id",
        "POSTMOTTAK_MAIL_FOLDER_FINISHED_ID": "Finished folder id",
        "POSTMOTTAK_MAIL_FOLDER_MANUALHANDLING_ID": "Manual handling folder id",
        "POSTMOTTAK_MAIL_FOLDER_ROBOT_LOG_NOMATCH_ID": "No match id",
        "POSTMOTTAK_MAIL_FOLDER_ROBOT_LOG_MAYBE_ID": "Maybe id",
        "POSTMOTTAK_MAIL_FOLDER_UNWANTED_RULE_ID": "unwanted rule id",
        "AZURE_OPENAI_API_KEY": "secret key",
        "AZURE_OPENAI_MODEL_NAME": "gpt-4o-mini",
        "AZURE_OPENAI_ENDPOINT": "sweden url",
        "AZURE_OPENAI_MAX_COMPLETION_TOKENS": "10000",
        "STATISTICS_BASE_URL": "stats url",
        "STATISTICS_KEY": "stats key",
        "EMAILTYPE_RF13.50_TEST_PROJECTNUMBER": "Denne trengs bare i dev og local og skal henvise til et test-prosjekt i testmiljøet",
        "EMAILTYPE_INNSYN_ADDRESSES": "bjarne.betjent@sesamstasjon.no",
        "EMAILTYPE_PENGETRANSPORTEN_FORWARD_ADDRESSES": "o.tidemann@sesamstasjon.no",
        "EMAILTYPE_LOYVEGARANTI_RESPONSIBLE_ENTERPRISE_RECNO": "81549300",
        "MONGODB_COLLECTION_NAME": "collection-name",
        "MONGODB_CONNECTION_STRING": "connection-string",
        "MONGODB_DATABASE_NAME": "database-name",
    }
}
```

## AI Provider Configuration

Postmottak-arkivering supports multiple AI providers for processing emails. You can configure which provider to use through environment variables. Choose either 'AzureOpenAI' or 'Mistral'

### Supported Providers

#### Azure OpenAI (Default)
To use Azure OpenAI, set the following environment variables:
```bash
AI_PROVIDER=AzureOpenAI
AZURE_OPENAI_MODEL_NAME=your-model
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MAX_COMPLETION_TOKENS=10000
```

#### Mistral AI
To use Mistral AI, set the following environment variables:
```bash
AI_PROVIDER=Mistral
MISTRAL_MODEL_NAME=your-model
MISTRAL_API_KEY=your-mistral-api-key
MISTRAL_BASE_URL=https://api.mistral.ai
MISTRAL_MAX_COMPLETION_TOKENS=10000
```

### Legge til nye e-posttyper

For å legge til nye e-posttyper, må du implementere en ny `IEmailType`-klasse som håndterer `MatchCriteria` og `HandledMessage` for den spesifikke e-posttypen.

Dersom du trenger/ønsker at e-posttypen skal sjekkes før andre e-posttyper, navngi filen med prefix på 3 tall for sortering, f.eks. `001_SomethingEmailType.cs`. Lavere tall har høyere prioritet i sorteringen.
