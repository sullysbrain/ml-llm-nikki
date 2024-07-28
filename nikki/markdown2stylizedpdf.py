import markdown2, os, weasyprint
from jinja2 import Template
from constants import LANGUAGE_CSS_FILE, LANGUAGE_PDF_PATH, LANGUAGE_LESSON_PATH

def convert_markdown_to_pdf(md_file, pdf_file, css_file):
    # Read Markdown file
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert Markdown to HTML
    # html_content = markdown.markdown(md_content)
    html_content = markdown2.markdown(md_content)

    # Read CSS file
    with open(css_file, 'r') as f:
        css_content = f.read()

    # Create a full HTML document
    html_template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>{{ css }}</style>
    </head>
    <body>
        {{ content }}
    </body>
    </html>
    ''')

    full_html = html_template.render(content=html_content, css=css_content)

    # Convert HTML to PDF
    weasyprint.HTML(string=full_html).write_pdf(pdf_file)




## USAGE

filename = 'ita_101_02_lessonplan'

md_file_import = os.path.join(LANGUAGE_LESSON_PATH, filename + '.md')
pdf_export = os.path.join(LANGUAGE_PDF_PATH, filename + '.pdf')

# Example usage
convert_markdown_to_pdf(md_file_import, pdf_export, LANGUAGE_CSS_FILE)
