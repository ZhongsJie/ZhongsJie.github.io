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

### 常用
```
$ hugo help

命令格式:
  hugo
  hugo [flags]
  hugo [command]
  hugo [command] [flags]

常用command:
  new         #为你的站点创建新的内容
  server      #一个高性能的web服务器

常用flags:
  -D, --buildDrafts                #包括被标记为draft的文章
  -E, --buildExpired               #包括已过期的文章
  -F, --buildFuture                #包括将在未来发布的文章

命令示例:
  hugo -D                          #生成静态文件并包括draft为true的文章
  hugo new post/new-content.md     #新建一篇文章
  hugo new site mysite             #新建一个称为mysite的站点
  hugo server --buildExpired       #启动服务器并包括已过期的文章
  hugo -e production               #构建
```

### New Post

```
hugo new posts/<YYYY>-<MM>-<DD>-<title>.md
```
### 目录说明
- archetypes: 储存.md的模板文件，类似于hexo的scaffolds，该文件夹的优先级高于主题下的/archetypes文件夹
- config.toml: 配置文件
- content: 储存网站的所有内容，类似于hexo的source
- data: 储存数据文件供模板调用，类似于hexo的source/_data
- layouts: 储存.html模板，该文件夹的优先级高于主题下的/layouts文件夹
- static: 储存图片,css,js等静态文件，该目录下的文件会直接拷贝到/public，该文件夹的优先级高于主题下的/static文件夹
- themes: 储存主题
- public: 执行hugo命令后，储存生成的静态文件

### Configure the site

```
baseURL = 'http://example.org/'
languageCode = 'en-us'
title = 'My New Hugo Site'
theme = 'ananke'
```
