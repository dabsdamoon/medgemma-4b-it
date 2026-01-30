# Local Development Notes

## Python Environment

Always use the conda environment `medgemma` when running Python commands:

```bash
conda run -n medgemma python <script.py>
```

Example:
```bash
conda run -n medgemma python run_pipeline.py --input old/png/circumcision.png --output output/test --mode full -v
```
