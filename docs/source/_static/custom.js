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

// functionality for Google Analytics dashboard
document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname.includes("dashboard.html")) {
        console.log("Google Analytics Dashboard Loaded");

        gapi.analytics.ready(function() {

            // Authorize the user using your Google Analytics View ID and API key
            gapi.analytics.auth.authorize({
                container: 'auth-button',  // You can create a div in your dashboard.md for this
                clientid: 'YOUR_CLIENT_ID',  // Replace with your Google API client ID
            });

            // Create a new Google Analytics view selector
            var viewSelector = new gapi.analytics.ViewSelector({
                container: 'view-selector'
            });

            // Fetch statistics such as sessions or pageviews
            viewSelector.on('change', function(ids) {
                var dataChart = new gapi.analytics.googleCharts.DataChart({
                    query: {
                        metrics: 'ga:sessions,ga:pageviews',
                        dimensions: 'ga:date',
                        'start-date': '30daysAgo',
                        'end-date': 'yesterday',
                        ids: ids
                    },
                    chart: {
                        container: 'analytics-dashboard',  // Ensure this ID matches the div in dashboard.md
                        type: 'LINE',
                        options: {
                            width: '100%'
                        }
                    }
                });

                dataChart.execute();
            });

            viewSelector.execute();
        });
    }
});
