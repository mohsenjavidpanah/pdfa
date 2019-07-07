Feature: Convert All Famouse Exported Programs To *.docx

    Scenario: Convert Google Docs Export To *.docs
        When Google Docs Exported PDF passed to program
        Then An Office *.docx File Must Be Generate
