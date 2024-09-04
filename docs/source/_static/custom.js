document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Custom JS loaded');
    
    // Copy to clipboard logic for all code blocks
    document.querySelectorAll('div.highlight').forEach((highlightDiv) => {
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

});
