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

### Step 6: Post-process LaTeX Formulas

After writing the Hugo post content, clean up LaTeX display math blocks to avoid Goldmark parsing conflicts:

**1. Remove blank lines inside `$$` blocks**

Goldmark's passthrough extension treats blank lines as paragraph breaks, which ends the passthrough block prematurely. Remove all empty/blank lines between opening `$$` and closing `$$`:

~~~
# BEFORE (broken):
$$
\begin{aligned}
\mathbf{c}_t^Q &= W^{DQ}\mathbf{h}_t, \\[4pt]

[\mathbf{q}_{t,1}^C;\cdots] &= \mathbf{q}_t^C
\end{aligned}
$$

# AFTER (fixed):
$$
\begin{aligned}
\mathbf{c}_t^Q &= W^{DQ}\mathbf{h}_t, \\[4pt]
[\mathbf{q}_{t,1}^C;\cdots] &= \mathbf{q}_t^C
\end{aligned}
$$
~~~

**2. Prevent standalone `=` on its own line**

A single `=` on its own line inside a `$$` block triggers Goldmark's Setext heading parser — the text above becomes an `<h1>`. Join `=` with the next line:

~~~
# BEFORE (broken):
$$
\operatorname{head}_{(i)}
=
\operatorname{softmax}(...)
$$

# AFTER (fixed):
$$
\operatorname{head}_{(i)}
= \operatorname{softmax}(...)
$$
~~~

Note: `\begin{aligned}` blocks using `&=` are safe — the `&` prefix prevents Setext detection.

**3. Add space between consecutive `$$` blocks**

When the closing `$$` of one block is immediately followed by the opening `$$` of the next block, HTML whitespace collapse produces `$$$$`, which KaTeX misparses as an empty open-close pair. Add a `<!-- -->` comment or extra blank line between them to force DOM separation:

~~~
# BEFORE (broken):
$$
q_{t,i} = W_i^Q · h_t, \quad k_t = W^K h_t
$$
$$
\operatorname{head}_{(i)} = \operatorname{softmax}(...)
$$

# AFTER (fixed):
$$
q_{t,i} = W_i^Q · h_t, \quad k_t = W^K h_t
$$

<!-- -->

$$
\operatorname{head}_{(i)} = \operatorname{softmax}(...)
$$
~~~

**4. Verify KaTeX auto-render config**

Ensure `layouts/partials/site-footer.html` includes all delimiter types:

```javascript
renderMathInElement(document.body, {
    delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\[", right: "\\]", display: true},
        {left: "$", right: "$", display: false},
        {left: "\\(", right: "\\)", display: false}
    ],
    throwOnError: false
});
```

**5. Prerequisite Hugo config**

Ensure `config.yaml` has passthrough extension and `unsafe` renderer:

```yaml
markup:
  goldmark:
    renderer:
      unsafe: true
    extensions:
      passthrough:
        delimiters:
          block:
            - ['$$', '$$']
            - ['\[', '\]']
          inline:
            - ['$', '$']
            - ['\(', '\)']
        enable: true
```

### Step 7: Save and Report

Report to the user:
- Source file location
- Destination file location (created by hugo new)
- List of images copied and their new locations
- List of internal links converted
- Any warnings or notes about unconverted elements

### Step 8: Start Hugo Server

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
- **Math/LaTeX**: Requires post-processing (see Step 6). Blank lines, standalone `=`, and consecutive `$$` blocks all cause Goldmark/KaTeX rendering failures. `\begin{aligned}` with `&=` is safe.
- **Dataview queries**: Warn user these won't work in Hugo
