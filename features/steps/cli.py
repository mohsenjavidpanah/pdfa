import os

from behave import when, then


os.chdir('./../..')


@when('Google Docs Exported PDF passed to program')
def google_docs_exported_pdf_passed_to_program(context):
    raise NotImplementedError(u'STEP: When Google Docs Exported PDF passed to program')


@then('An Office *.docx File Must Be Generate')
def an_office_x_docx_file_must_be_generate(context):
    raise NotImplementedError(u'STEP: Then An Office *.docx File Must Be Generate')
