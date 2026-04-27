# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hugo-based static blog deployed to GitHub Pages. The blog uses the Ananke theme and is written in Chinese.

## Common Commands

```bash
# Create a new blog post
hugo new posts/<YYYY>-<MM>-<DD>-<title>.md

# Start local dev server (includes drafts)
hugo server -D

# Build for production
hugo
```

The blog is served at `http://localhost:1313/` when running the dev server.

## Content Structure

- **Posts**: `content/posts/` - Blog articles in Markdown format
- **Images**: `static/images/` - Static assets organized by date (e.g., `static/images/2025-12/`)
- **Theme**: `themes/ananke/` - Hugo Ananke theme (submodule)
- **Config**: `config.yaml` - Site configuration

## Frontmatter Format

Blog posts use this frontmatter:

```yaml
---
title: "Post Title"
date: 2025-12-28T21:15:37+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags: []
categories: []
showToc: false
TocOpen: false
disableShare: true
cover:
    image: "/images/2025-12/image.png"
    caption: ""
    alt: ""
    relative: false
---
```

## Publishing

The site is built to `docs/` directory for GitHub Pages deployment. Commit and push to trigger deployment.

## Custom Skills

- `.claude/skills/obsidian-to-hugo/` - Converts Obsidian markdown notes to Hugo blog format
