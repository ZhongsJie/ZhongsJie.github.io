# ZhongsJie'Blog


## Github Pages

settings > Pages

## Hugo

> Static site [Hugo](https://gohugo.io/)

```
hugo new site quickstart
cd quickstart
git init
git submodule add https://github.com/theNewDynamic/gohugo-theme-ananke themes/ananke
echo "theme = 'ananke'" >> config.toml
hugo server
```
### New Post

```
hugo new posts/<YYYY>-<MM>-<DD>-<title>.md
```
### Preview

`hugo server -D`

### Build

`hugo -e production`

### Configure the site

```
baseURL = 'http://example.org/'
languageCode = 'en-us'
title = 'My New Hugo Site'
theme = 'ananke'
```
