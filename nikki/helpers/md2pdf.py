import markdown2, os, weasyprint, argparse
from datetime import datetime
from jinja2 import Template
from constants import LANGUAGE_CSS_FILE, LANGUAGE_PDF_PATH, LANGUAGE_LESSON_PATH


# CLI Parser for Embed_docs
# usage: to load reports, type in CLI:  python embed_docs.py reports
# usage: to load tutor, type in CLI:  python embed_docs.py tutor

parser = argparse.ArgumentParser()
parser.add_argument("docs")
args = parser.parse_args()

doc2convert = args.docs
print(f"Converting {doc2convert} to PDF")


def convert_markdown_to_pdf(md_file, pdf_file, css_file):
    # Read Markdown file
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert Markdown to HTML
    # html_content = markdown.markdown(md_content)
    html_content = markdown2.markdown(md_content)

    # Add a copyright watermark to the PDF
    now = datetime.now() # current date and time
    year = now.strftime("%Y")
    copyright = "© " + year + " Digital Blacksmiths, Nicoletta Carino. All Rigths Reserved."

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
        <style>
            {{ css }}
            @page {
                size: auto;
                margin: 30px;
            }
        </style>
    </head>
    <body>
        <h1 class="header" style="color: #e6e6e6; padding-top:20px;font-family: 'Gill Sans';font-style: semi-bold;margin;text-align:center;font-size:55px;line-height:0.9em;">Italian<br>With Nikki</h1>
        <div class="content">
            {{ content }}
        </div>
        <div class="watermark">{{ copyright }}</div>
    </body>
    </html>
    ''')

    full_html = html_template.render(content=html_content, css=css_content, copyright=copyright)

    # Convert HTML to PDF
    weasyprint.HTML(string=full_html, base_url=".").write_pdf(pdf_file)





## USAGE

# filename = 'ita_101_02_lessonplan'
filename = doc2convert

md_file_import = os.path.join(LANGUAGE_LESSON_PATH, filename + '.md')
pdf_export = os.path.join(LANGUAGE_PDF_PATH, filename + '.pdf')

# Example usage
convert_markdown_to_pdf(md_file_import, pdf_export, LANGUAGE_CSS_FILE)
