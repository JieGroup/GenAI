document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Custom JS loaded');

    // Copy to clipboard logic for code blocks (highlight-default)
    document.querySelectorAll('div.highlight-default').forEach((highlightDiv) => {
        console.log('Processing div.highlight-default (code block):', highlightDiv);

        const pre = highlightDiv.querySelector('pre');
        if (pre) {
            console.log('Found code block:', pre);
            const button = document.createElement('button');
            button.className = 'md-clipboard md-icon';
            button.title = 'Copy to clipboard';
            button.setAttribute('data-clipboard-target', `#${highlightDiv.id} > code`);
            highlightDiv.style.position = 'relative';  // Ensure the highlight div is positioned relatively
            highlightDiv.insertBefore(button, pre);

            button.addEventListener('click', () => {
                console.log('Copy button clicked');
                const range = document.createRange();
                range.selectNode(pre);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);

                try {
                    const successful = document.execCommand('copy');
                    if (successful) {
                        button.textContent = 'Copied!';
                        console.log('Copy successful');
                        setTimeout(() => {
                            button.textContent = '';
                        }, 2000);
                    } else {
                        button.textContent = 'Failed';
                        console.error('Copy failed');
                    }
                } catch (err) {
                    button.textContent = 'Failed';
                    console.error('Error during copy:', err);
                }

                window.getSelection().removeAllRanges();
            });
        }
    });

    // Initialize Mermaid diagrams after the page loads (highlight-mermaid), only for logging purposes
    document.querySelectorAll('div.highlight-mermaid').forEach((mermaidDiv) => {
        console.log('Processing Mermaid diagram:', mermaidDiv);
    });

    if (typeof mermaid !== 'undefined') {
        console.log('Mermaid found. Initializing Mermaid...');
        mermaid.initialize({ startOnLoad: true });
        console.log('Mermaid initialized');
    } else {
        console.error('Mermaid is not loaded');
    }

    // Initialize MathJax after the page loads (using MathJax v3 API)
    if (typeof MathJax !== 'undefined') {
        console.log('MathJax found. Configuring MathJax v3...');
        
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],  // Handle inline math with $...$
                displayMath: [['$$', '$$'], ['\\[', '\\]']],  // Handle block math with $$...$$
                processEscapes: true  // Escape special characters
            },
            svg: {
                fontCache: 'global'
            }
        };

        MathJax.typesetPromise().then(() => {
            console.log('MathJax typeset complete');
        }).catch((err) => {
            console.error('MathJax typeset failed: ', err);
        });
    } else {
        console.error('MathJax is not loaded');
    }
});
