document.addEventListener('DOMContentLoaded', function () {
    const links = document.querySelectorAll('.page__content a');
    const tooltip = document.createElement('div');
    tooltip.id = 'link-preview-tooltip';
    document.body.appendChild(tooltip);

    links.forEach(link => {
        // Only target external links that start with http and don't have the 'no-preview' class
        if (link.hostname !== window.location.hostname && link.href.startsWith('http') && !link.classList.contains('no-preview')) {
            link.addEventListener('mouseenter', function (e) {
                const url = link.href;
                tooltip.style.display = 'block';
                tooltip.classList.add('loading');
                tooltip.style.backgroundImage = 'none'; // Clear previous image

                // Using Microlink API with specific viewport for desktop view
                const imageUrl = `https://api.microlink.io/?url=${encodeURIComponent(url)}&screenshot=true&meta=false&embed=screenshot.url&viewport.width=1280&viewport.height=800&viewport.deviceScaleFactor=1`;

                // Preload image
                const img = new Image();
                img.onload = function () {
                    tooltip.style.backgroundImage = `url(${imageUrl})`;
                    tooltip.classList.remove('loading');
                };
                img.src = imageUrl;

                // Initial positioning
                positionTooltip(e, tooltip);
            });

            link.addEventListener('mousemove', function (e) {
                positionTooltip(e, tooltip);
            });

            link.addEventListener('mouseleave', function () {
                tooltip.style.display = 'none';
                tooltip.style.backgroundImage = 'none';
            });
        }
    });

    function positionTooltip(e, tooltip) {
        const offset = 15;
        let top = e.pageY + offset;
        let left = e.pageX + offset;

        // Check for right edge overflow
        if (left + 300 > window.innerWidth) {
            left = e.pageX - 315; // 300 width + offset
        }

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    }
});
