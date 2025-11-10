# GitHub Pages Setup for Presto-Viz (Cross-Repository Usage)

This guide shows how to use presto-viz as a tool in your LMR2 repository and automatically deploy visualization output to GitHub Pages.

## Overview

**Scenario**: You have two repositories:
- **presto-viz**: Tool repository containing visualization scripts
- **LMR2**: Your data repository where you want to publish visualizations

**Goal**: Run presto-viz scripts in your LMR2 workflow and commit the output to LMR2's `docs/` folder for GitHub Pages.

## How It Works

Script 3 (`3_make_html_file.py`) can automatically:
1. Copy visualization output from the temporary output directory to your target repository
2. Commit the changes to your specified repository (LMR2)
3. Push to GitHub Pages branch

This is all controlled via environment variables.

## Environment Variables

### Required
- **`COMMIT_VIZ_OUTPUT`**: Set to `true` to enable git commit functionality
- **`GIT_REPO_ROOT`**: **REQUIRED** - Absolute path to the repository where you want to commit (your LMR2 repo)

### Optional (with defaults)
- **`VIZ_GIT_BRANCH`**: Branch to commit and push to
  - Default: `gh-pages`
  - For GitHub Pages from main branch, use: `main`

- **`VIZ_OUTPUT_PATH`**: Path within target repo where viz files should be placed
  - Default: `docs` (recommended for GitHub Pages)
  - Alternative: `viz`, or any custom path

- **`VIZ_COMMIT_MSG`**: Custom commit message
  - Default: `Update visualization: {dataset_txt} v{version_txt}`

## Complete GitHub Actions Workflow Example

Here's a complete workflow for your **LMR2** repository:

```yaml
name: Generate CFR Visualization and Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write  # Required to push commits

jobs:
  generate-and-deploy:
    runs-on: ubuntu-latest

    steps:
      # 1. Checkout your LMR2 repository (where output will be committed)
      - name: Checkout LMR2 repository
        uses: actions/checkout@v4
        with:
          ref: main  # Or gh-pages if using separate branch
          fetch-depth: 0  # Get full history for proper commits

      # 2. Checkout presto-viz as a tool (in a subdirectory)
      - name: Checkout presto-viz tools
        uses: actions/checkout@v4
        with:
          repository: DaveEdge1/presto-viz
          path: presto-viz  # Clone into subdirectory

      # 3. Setup Python environment
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      # 4. Install dependencies
      - name: Install dependencies
        run: |
          pip install numpy xarray matplotlib cartopy regionmask bokeh netCDF4 pyyaml

      # 5. Configure git for commits (important!)
      - name: Configure git
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "actions@github.com"

      # 6. Run the visualization scripts with GitHub Pages commit enabled
      - name: Run presto-viz scripts
        env:
          # Enable automatic git commit to LMR2 repo
          COMMIT_VIZ_OUTPUT: 'true'

          # Point to LMR2 repo (current repo, not presto-viz)
          GIT_REPO_ROOT: ${{ github.workspace }}

          # Where to place files in LMR2 repo (docs/ for GitHub Pages)
          VIZ_OUTPUT_PATH: 'docs'

          # Which branch to commit to (main or gh-pages)
          VIZ_GIT_BRANCH: 'main'

          # Custom commit message (optional)
          VIZ_COMMIT_MSG: 'Update CFR visualization - Run ${{ github.run_number }}'
        run: |
          # Define paths
          DATA_DIR="${{ github.workspace }}/data/CFR_Run_17_Reviz_22"
          OUTPUT_DIR="${{ github.workspace }}/output/CFR_Run_17_Reviz_22"
          WEB_DATA_DIR="${{ github.workspace }}/presto-viz/viz/web_assets/"

          # Run the three scripts
          cd presto-viz
          python 1_format_data_daholocene_graphem.py "$DATA_DIR"
          python 2_make_maps_and_ts.py "$DATA_DIR" "$OUTPUT_DIR"
          python 3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "$WEB_DATA_DIR"

          # Note: Script 3 will automatically commit to LMR2 repo and push

      # 7. Optional: Create a summary
      - name: Job summary
        if: success()
        run: |
          echo "✅ Visualization generated and deployed to GitHub Pages" >> $GITHUB_STEP_SUMMARY
          echo "📊 View at: https://DaveEdge1.github.io/LMR2/" >> $GITHUB_STEP_SUMMARY
```

## GitHub Pages Configuration in LMR2

After setting up the workflow, configure GitHub Pages in your **LMR2** repository:

1. Go to **Settings** → **Pages**
2. Under **Source**, select:
   - **Branch**: `main`
   - **Folder**: `/docs`
3. Click **Save**

Your visualization will be available at: `https://DaveEdge1.github.io/LMR2/`

## Alternative: Using a Separate gh-pages Branch

If you prefer to keep visualization separate from your main branch:

```yaml
# In your workflow, change these settings:
env:
  VIZ_GIT_BRANCH: 'gh-pages'
  VIZ_OUTPUT_PATH: '.'  # Root of gh-pages branch

# And in the checkout step:
- name: Checkout LMR2 repository
  uses: actions/checkout@v4
  with:
    ref: gh-pages  # Checkout gh-pages branch
    fetch-depth: 0
```

Then configure GitHub Pages to serve from `gh-pages` branch root (`/`).

## Alternative: Manual Git Operations (Without Built-in Commit)

If you prefer to handle git operations manually in your workflow:

```yaml
# Disable automatic commit
env:
  COMMIT_VIZ_OUTPUT: 'false'  # or omit entirely

# Then add these steps after running scripts:
- name: Copy visualization to docs
  run: |
    mkdir -p docs
    cp -r output/CFR_Run_17_Reviz_22/viz/* docs/

- name: Commit and push
  run: |
    git config user.name "GitHub Actions Bot"
    git config user.email "actions@github.com"
    git add docs/
    git commit -m "Update visualization" || echo "No changes to commit"
    git push origin main
```

## Local Testing

To test the cross-repository commit functionality locally:

```bash
# Clone both repos
git clone https://github.com/DaveEdge1/LMR2.git
git clone https://github.com/DaveEdge1/presto-viz.git

cd presto-viz

# Set environment variables pointing to LMR2 repo
export COMMIT_VIZ_OUTPUT=true
export GIT_REPO_ROOT=/absolute/path/to/LMR2
export VIZ_OUTPUT_PATH=docs
export VIZ_GIT_BRANCH=main

# Run scripts
python 1_format_data_daholocene_graphem.py "/path/to/data"
python 2_make_maps_and_ts.py "/path/to/data" "/path/to/output"
python 3_make_html_file.py "/path/to/data" "/path/to/output" "./viz/web_assets/"

# Check LMR2 repo for changes
cd /path/to/LMR2
git status  # Should show changes in docs/
```

## Directory Structure

Your workflow will create this structure:

```
LMR2/                          # Your data repository
├── .github/
│   └── workflows/
│       └── visualize.yml      # Your workflow file
├── data/
│   └── CFR_Run_17_Reviz_22/   # Input data
├── output/                     # Temporary output (not committed)
│   └── CFR_Run_17_Reviz_22/
│       └── viz/               # Generated by script 2
└── docs/                       # Committed to git for GitHub Pages
    ├── visualizer.html        # Copied here by script 3
    ├── assets/
    │   └── lmr/
    │       ├── map_*.png
    │       └── ts_*.html
    └── ...

presto-viz/                     # Tool repository (checked out in workflow)
├── 1_format_data_daholocene_graphem.py
├── 2_make_maps_and_ts.py
└── 3_make_html_file.py
```

## Troubleshooting

**Issue**: "ERROR: GIT_REPO_ROOT environment variable must be set"
- **Solution**: Ensure `GIT_REPO_ROOT` is set in your workflow's `env:` section

**Issue**: "Git push failed"
- **Solution**: Check that your workflow has `permissions: contents: write`
- **Solution**: Ensure branch protection rules allow GitHub Actions to push

**Issue**: "Files not showing on GitHub Pages"
- **Solution**: Check GitHub Pages settings point to correct branch and folder
- **Solution**: Ensure `docs/` folder contains `visualizer.html` and `assets/`
- **Solution**: Check that workflow actually committed files: look at commit history

**Issue**: "Visualization works locally but not in workflow"
- **Solution**: Verify all paths use `${{ github.workspace }}` for absolute paths
- **Solution**: Check that data files exist in expected locations

## Key Differences from Single-Repo Setup

1. **Two repositories**: presto-viz (tool) and LMR2 (data + output)
2. **GIT_REPO_ROOT is REQUIRED**: Must explicitly point to LMR2
3. **Checkout order matters**: Check out LMR2 first (target), then presto-viz (tool)
4. **Git config in LMR2**: Configure git in the repository where you want to commit
5. **Paths are cross-repo**: Scripts run from presto-viz but commit to LMR2

This approach keeps your tool repository (presto-viz) separate from your data repository (LMR2) while still enabling automatic GitHub Pages deployment.
