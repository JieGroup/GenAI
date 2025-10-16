#!/usr/bin/env python3
"""
Simple README.md preview tool
"""
import os
import webbrowser
import tempfile

def create_preview():
    # Read README.md
    with open('README.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create HTML with embedded markdown parser
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GenAI README Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            line-height: 1.6;
            background-color: #ffffff;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ 
            border: 1px solid #e1e4e8; 
            padding: 12px 16px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #f6f8fa; 
            font-weight: 600;
            color: #24292e;
        }}
        tr:nth-child(even) {{
            background-color: #f6f8fa;
        }}
        code {{ 
            background-color: #f6f8fa; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-family: 'SFMono-Regular', 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }}
        pre {{ 
            background-color: #f6f8fa; 
            padding: 16px; 
            border-radius: 6px; 
            overflow-x: auto; 
            border: 1px solid #e1e4e8;
            font-family: 'SFMono-Regular', 'Monaco', 'Menlo', monospace;
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 4px;
        }}
        .center {{ 
            text-align: center; 
        }}
        h1, h2, h3, h4, h5, h6 {{ 
            color: #24292e; 
            margin-top: 30px;
            margin-bottom: 16px;
        }}
        h1 {{ border-bottom: 1px solid #e1e4e8; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; }}
        blockquote {{
            border-left: 4px solid #0366d6;
            margin: 20px 0;
            padding: 0 16px;
            background-color: #f6f8fa;
            color: #586069;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            height: 1px;
            background-color: #e1e4e8;
            margin: 30px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div id="content">Loading...</div>
    <script>
        const markdown = `{markdown_content}`;
        document.getElementById('content').innerHTML = marked.parse(markdown);
    </script>
</body>
</html>"""
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_template)
        temp_file = f.name
    
    print(f"✅ Created README preview: {temp_file}")
    print("🌐 Opening in your default browser...")
    
    # Open in browser
    webbrowser.open(f'file://{temp_file}')
    
    return temp_file

if __name__ == "__main__":
    if os.path.exists('README.md'):
        create_preview()
    else:
        print("❌ README.md not found in current directory")
