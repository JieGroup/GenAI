document.addEventListener('DOMContentLoaded', (event) => {
    console.log('Custom JS loaded');
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
                const range = document.createRange();
                range.selectNode(pre);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);

                try {
                    const successful = document.execCommand('copy');
                    if (successful) {
                        button.textContent = 'Copied!';
                        setTimeout(() => {
                            button.textContent = '';
                        }, 2000);
                    } else {
                        button.textContent = 'Failed';
                    }
                } catch (err) {
                    button.textContent = 'Failed';
                }

                window.getSelection().removeAllRanges();
            });
        }
    });
});
