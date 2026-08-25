# Stage 1 sampling smoke test

This example provides three small synthetic trajectories for validating dataset loading, frame sampling, and Stage 1 artifact generation without calling a remote model.

From the repository root, set the configuration placeholders and run sampling:

```bash
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
python -m prmeval.cli sample \
  --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli validate-samples \
  --samples evaluation_output/jsonl-progress-full-smoke/samples.jsonl
```

Stage 1 does not send requests or require `OPENAI_API_KEY`; `BASE_URL` and `MODEL_ID` are only used to resolve the shared configuration.
