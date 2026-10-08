FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html
COPY retrieval.html /usr/share/nginx/html/retrieval.html
COPY llm.html /usr/share/nginx/html/llm.html
COPY evaluation.html /usr/share/nginx/html/evaluation.html
COPY styles.css /usr/share/nginx/html/styles.css
COPY script.js /usr/share/nginx/html/script.js
COPY portals.css /usr/share/nginx/html/portals.css
COPY portals.js /usr/share/nginx/html/portals.js
COPY evaluation.css /usr/share/nginx/html/evaluation.css
COPY evaluation.js /usr/share/nginx/html/evaluation.js

EXPOSE 80
