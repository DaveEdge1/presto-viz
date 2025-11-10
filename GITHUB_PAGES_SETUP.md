# GitHub Pages Setup for Presto-Viz

This document explains how to configure script 3 to automatically commit visualization output to your repository for GitHub Pages deployment.

## How It Works

Script 3 (`3_make_html_file.py`) now includes optional git commit functionality that can be enabled via environment variables. When enabled, it will:

1. Copy the visualization output to a specified location in your repo
2. Stage the changes with `git add`
3. Commit the changes with a custom message
4. Push to a specified branch (default: `gh-pages`)

## Environment Variables

Configure the behavior using these environment variables:

### Required
- **`COMMIT_VIZ_OUTPUT`**: Set to `true` to enable git commit functionality
  - Default: `false` (disabled)

### Optional (with defaults)
- **`VIZ_GIT_BRANCH`**: Branch to commit and push to
  - Default: `gh-pages`

- **`VIZ_COMMIT_MSG`**: Custom commit message
  - Default: `Update visualization: {dataset_txt} v{version_txt}`

- **`VIZ_OUTPUT_PATH`**: Path within repo where viz files should be placed
  - Default: `viz`

- **`GIT_REPO_ROOT`**: Absolute path to git repository root
  - Default: Parent directory of `output_dir`

## GitHub Actions Workflow Example

Here's how to configure this in your LMR2 repository's GitHub Actions workflow:

```yaml
name: Generate and Deploy Visualization

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  generate-viz:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout LMR2 repo
        uses: actions/checkout@v4
        with:
          path: LMR2

      - name: Checkout presto-viz repo
        uses: actions/checkout@v4
        with:
          repository: DaveEdge1/presto-viz
          path: presto-viz

      - name: Setup Python and dependencies
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install numpy xarray matplotlib cartopy regionmask bokeh netCDF4

      - name: Run visualization scripts
        working-directory: presto-viz
        env:
          # Enable git commit functionality
          COMMIT_VIZ_OUTPUT: 'true'

          # Configure where to commit (use LMR2 repo for GitHub Pages)
          GIT_REPO_ROOT: ${{ github.workspace }}/LMR2
          VIZ_OUTPUT_PATH: 'docs'  # GitHub Pages serves from docs/ folder
          VIZ_GIT_BRANCH: 'main'   # Or 'gh-pages' if using separate branch

          # Optional: custom commit message
          VIZ_COMMIT_MSG: 'Update CFR visualization - ${{ github.sha }}'
        run: |
          DATA_DIR="${{ github.workspace }}/LMR2/data/CFR_Run_17_Reviz_22"
          OUTPUT_DIR="${{ github.workspace }}/LMR2/output/CFR_Run_17_Reviz_22"
          WEB_DATA_DIR="${{ github.workspace }}/presto-viz/viz/web_assets/"

          # Run the scripts
          python 1_format_data_daholocene_graphem.py "$DATA_DIR"
          python 2_make_maps_and_ts.py "$DATA_DIR" "$OUTPUT_DIR"
          python 3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "$WEB_DATA_DIR"

      - name: Configure git for commit
        working-directory: LMR2
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "actions@github.com"

      # Note: The commit and push happens inside script 3
      # If you need to handle authentication differently, you can
      # disable COMMIT_VIZ_OUTPUT and handle git operations here instead

```

## Alternative: Manual Git Operations in Workflow

If you prefer to handle git operations in your workflow file instead of script 3, keep `COMMIT_VIZ_OUTPUT=false` and add these steps:

```yaml
      - name: Copy visualization output
        run: |
          mkdir -p LMR2/docs
          cp -r LMR2/output/CFR_Run_17_Reviz_22/viz/* LMR2/docs/

      - name: Commit and push to GitHub Pages
        working-directory: LMR2
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "actions@github.com"
          git add docs/
          git commit -m "Update visualization from run ${{ github.run_number }}" || echo "No changes to commit"
          git push origin main
```

## GitHub Pages Configuration

After setting this up, configure GitHub Pages in your LMR2 repository:

1. Go to **Settings** → **Pages**
2. Under **Source**, select:
   - **Branch**: `main` (or `gh-pages` if using separate branch)
   - **Folder**: `/docs` (or `/` if `VIZ_OUTPUT_PATH='viz'` at repo root)
3. Click **Save**

Your visualization will be available at: `https://DaveEdge1.github.io/LMR2/`

## Local Testing

To test the git commit functionality locally:

```bash
export COMMIT_VIZ_OUTPUT=true
export GIT_REPO_ROOT=/path/to/your/repo
export VIZ_OUTPUT_PATH=docs
export VIZ_GIT_BRANCH=main

python 3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "$WEB_DATA_DIR"
```

## Security Notes

- The script uses `os.system()` for git commands. Ensure commit messages and paths are properly sanitized.
- In GitHub Actions, the default `GITHUB_TOKEN` has permissions to push to the repository.
- For pushing to different repositories, configure a Personal Access Token (PAT) with appropriate permissions.
