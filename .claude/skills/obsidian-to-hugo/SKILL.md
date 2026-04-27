---
name: obsidian-to-hugo
description: Convert Obsidian markdown notes to Hugo blog format. Use this skill whenever the user wants to import an Obsidian note to their Hugo-based blog. The skill handles image link conversion, internal link transformation, frontmatter generation, and proper file organization by date. TRIGGER this skill when the user mentions importing, converting, or copying notes from Obsidian to their blog, or when they want to migrate Obsidian content to Hugo.
---

# Obsidian to Hugo Blog Converter

This skill converts Obsidian markdown notes to Hugo blog format. It handles image link conversion, internal wiki-link transformation, frontmatter generation, and proper file organization.

## Input Requirements

1. **Obsidian file path**: The full path to the Obsidian markdown file
   - If not provided, ask the user: "Please provide the full path to the Obsidian markdown file you want to convert"
2. **Target date**: The publication date for Hugo frontmatter (format: YYYY-MM-DD)
   - If not provided, ask the user

## Conversion Process

### Step 1: Read the Obsidian File

Read the source markdown file from the Obsidian vault. Preserve the original file untouched.

### Step 2: Create Hugo Post

Use Hugo's built-in command to create a new post:

```bash
hugo new posts/{YYYY-MM-DD}-{slugified-title}.md
```

This creates a new file in `content/posts/` with the proper frontmatter template.

### Step 3: Extract and Copy Images

1. Find all image references in the Obsidian markdown:
   - Wiki-style: `![[image.png]]`
   - Markdown-style: `![](image.png)` or `![](assets/image.png)`
   - Absolute paths: `![](C:/Users/.../image.png)`

2. For each image:
   - Extract the filename
   - Copy the image to `static/images/{YYYY-MM}/`
   - Rename if necessary to avoid conflicts

3. Update image references in the markdown to:
   ```
   ![](/images/{YYYY-MM}/{filename})
   ```

### Step 4: Convert Internal Links

Find and convert Obsidian wiki-links:

1. `[[note-name]]` → `[note-name](./note-name.md)`
2. `[[note-name|display text]]` → `[display text](./note-name.md)`

For links that point to content that exists in the blog, keep the relative link. For links to non-existent notes, keep the link but note it may need manual fixing.

### Step 5: Update Hugo Post

Read the newly created Hugo file, then:

1. Extract the title from the Obsidian file:
   - First H1 heading (`# Title`)
   - First H2 heading if no H1
   - Filename (without extension) as fallback

2. Update the frontmatter with the extracted title:
   ```yaml
   ---
   title: "{extracted title}"
   date: {YYYY-MM-DD}T{current time}+08:00
   draft: true
   subtitle: ""
   author: "ZhongsJie"
   tags: []
   categories: []
   showToc: false
   TocOpen: false
   disableShare: true
   cover:
       image: ""
       caption: ""
       alt: ""
       relative: false
   ---
   ```

3. Replace the content after frontmatter with the converted Obsidian content

### Step 6: Save and Report

Report to the user:
- Source file location
- Destination file location (created by hugo new)
- List of images copied and their new locations
- List of internal links converted
- Any warnings or notes about unconverted elements

### Step 7: Start Hugo Server

Run the Hugo development server to preview the blog:

```bash
hugo server -D
```

Run this in the background so the user can view the blog at `http://localhost:1313/`.

## Example

**Input**: An Obsidian note from `/Users/zhongsjie/Library/Mobile Documents/iCloud~md~obsidian/Documents/Citadel-Notebooks/notes/my-note.md`

**Output**:
- Hugo post created: `content/posts/2025-04-27-my-note.md`
- Images copied to: `static/images/2025-04/`
- Wiki-links converted to markdown links
- Hugo server started at `http://localhost:1313/`

## Edge Cases

- **No images**: Skip image processing steps
- **External links**: Leave unchanged (they work in Hugo)
- **Code blocks**: Leave unchanged
- **Math/LaTeX**: Leave unchanged (Hugo may need MathJax/KaTeX configuration)
- **Dataview queries**: Warn user these won't work in Hugo
