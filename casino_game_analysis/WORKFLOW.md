# Casino Game Analysis App Workflow

This document outlines the complete workflow for maintaining and updating the casino game semantic search application, from generating summaries to updating the search index.

## Complete Workflow Overview

![Workflow Diagram](https://mermaid.ink/img/pako:eNp1kk9rwkAQxb_KMGcTSCRpLHjStmCxB00vPe4lbjdxMH-c3YQkIn73blJTQ2m7p-G9-c17s0fZtAalkkVV7PtSd63wvZc8m5YwVVbkdU5YVpXjnFtVaVVshYEsCqINGP-vkRczkJJprh1oVFVbAN9CnCarh-Qxjt8SyJPNOmuYcD73k3STz2Oc-GRnQisE17xjXLyVqFWFpjGwu2HsvYGl-aiV_YKWZu_oWil7uUPdglUkx9CXrta09MZVXIAbG-O8hkNjUK2FvLUFm9Y0FfHVbXx76y-ieB7PcBqnyF8r3P_WvPX_VOymngF3UjhZXnDuahRHvuB6DbhDXxcS3_LhuWYa3XXgTGNpz6TT8h9cV8ChbiGQM_kqZXOBzBO_Z81ZSoYmM6NWUDfUHoRyM2XJQYYtKK1FnffqH5LSB-3e51Ps0lqKSSc35nJZWeweT6apGkVxEvJN9r4chn5rds9V-e_vL9s31tg?type=png)

## 1. Summary Generation

The first step is to generate structured summaries for games that don't already have them.

### Commands

```bash
# Change to scripts directory
cd /path/to/windsurf/casino_game_analysis/scripts

# Estimate cost before generating summaries
python generate_reliable_summaries.py --estimate-only

# Generate a specific number of summaries (e.g., 1000)
python generate_reliable_summaries.py -n 1000

# Generate summaries using a different model
python generate_reliable_summaries.py -m gpt-4o-mini -n 500

# Resume from where previous generation left off
python generate_reliable_summaries.py
```

### Best Practices

- **Start with Estimation**: Always run with `--estimate-only` first to check costs
- **Batch Processing**: Process 500-1000 summaries at a time to avoid timeouts
- **Model Selection**: Use `gpt-4.1-mini` for best balance of cost and quality
- **Monitoring**: Check the progress logs periodically

## 2. Embedding Generation

After generating summaries, create embeddings to make the games searchable.

### Commands

```bash
# Generate embeddings for all summaries
python generate_summary_embeddings.py

# Generate embeddings with custom input/output files
python generate_summary_embeddings.py --input custom_summaries.csv --output custom_embeddings.csv
```

### Best Practices

- **Regular Updates**: Run this after each summary generation batch
- **Monitoring**: Check the logs for any errors
- **API Costs**: Embeddings are much cheaper than summary generation

## 3. Search App Update

After generating embeddings, update the search app to use the latest data.

### Commands

```bash
# Update the search app with the latest embeddings
python update_search_app.py

# Force update even if no changes are detected
python update_search_app.py --force
```

## 4. Analysis and Diagnostics

Periodically analyze your data to understand coverage and quality.

### Commands

```bash
# Analyze summary coverage
python analyze_name_summaries.py

# Check for filtering issues
python analyze_game_filtering.py
```

## 5. Complete Regeneration (If Needed)

If you need to regenerate everything from scratch:

```bash
# Step 1: Generate all summaries
python generate_reliable_summaries.py --override

# Step 2: Regenerate all embeddings
python generate_summary_embeddings.py

# Step 3: Update the search app
python update_search_app.py --force
```

## Workflow for Adding New Games

When you have new casino games to add to your dataset:

1. Add the games to your base CSV file (`bigwinboard_cleaned.csv`)
2. Run `python generate_reliable_summaries.py` to generate summaries for only the new games
3. Run `python generate_summary_embeddings.py` to create embeddings for all summaries
4. Run `python update_search_app.py` to update the search app with the new data

## Troubleshooting

### API Key Issues

```bash
# Check if your API key is being detected
echo $OPENAI_API_KEY

# Set the API key temporarily if needed
export OPENAI_API_KEY=your-key-here
```

### Incomplete Processing

If summary generation stops unexpectedly:
- The script saves progress automatically
- Simply rerun the same command to resume

### Embedding Errors

If you encounter errors during embedding generation:
- Check the log file for details
- Try clearing temporary checkpoints and rerunning the script

### Search App Not Updating

If the search app isn't showing new content:
- Verify that `update_search_app.py` ran successfully
- Check if the app needs to be restarted to pick up changes

## Search App UI Considerations

Remember that your search app has specific UI requirements:
- Dark theme with pronounced purple gradients
- Very dark card backgrounds (#1e1e2a) with prominent shadows
- Developer names shown without a prefix
- Volatility shown with "Volatility:" prefix
- Custom scrollbar with gradient styling

All updated data will automatically maintain these styling preferences when properly integrated with the search API.

## Performance Statistics

- **Typical Processing Times**:
  - Summary generation: ~1.5 games/second with GPT-4.1 Mini
  - Embedding generation: ~5 games/second
- **API Cost Examples**:
  - 1000 summaries with GPT-4.1 Mini: ~$1.11
  - Embeddings for 1000 games: ~$0.05

## Recommended Maintenance Schedule

- Weekly: Generate summaries for new games
- Weekly: Run `generate_summary_embeddings.py` and `update_search_app.py` after new summaries
- Monthly: Check for any games with missing summaries or embeddings
- Quarterly: Run full diagnostics to ensure complete coverage

---

Last updated: May 9, 2025
