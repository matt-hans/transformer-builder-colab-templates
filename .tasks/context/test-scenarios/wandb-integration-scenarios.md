# W&B Integration Test Scenarios

## Scenario Category: Initial Setup

### TS-001: First-Time User Setup
```
Given: User has never used W&B before
  And: User opens training.ipynb in Google Colab
When: They reach the W&B login cell
Then: Clear instructions displayed with signup link
  And: Prompted for API key via wandb.login()
  And: After entering key, sees "✅ W&B logged in successfully"
  And: Training proceeds with experiment tracking enabled
```

### TS-002: Returning User Auto-Login
```
Given: User previously saved WANDB_API_KEY in Colab Secrets
When: They run training.ipynb
Then: Auto-login succeeds without manual key entry
  And: No interruption to training flow
  And: Confirmation: "✅ W&B logged in via Colab Secrets"
```

### TS-003: Offline Mode Fallback
```
Given: User skips W&B login cell
  Or: Internet connectivity unavailable
When: Training starts with wandb.init()
Then: Warning displayed: "⚠️ W&B tracking disabled, running offline mode"
  And: Training continues without errors
  And: Metrics logged to local .wandb/ directory
  And: Instructions provided: "Run 'wandb sync .wandb/' to upload later"
```

## Scenario Category: Experiment Tracking

### TS-004: Hyperparameter Logging
```
Given: Training config with learning_rate=5e-5, batch_size=4, epochs=10
When: wandb.init() called with these parameters
Then: W&B dashboard config tab shows all hyperparameters
  And: Model metadata included (vocab_size, total_params, model_type)
  And: Environment info logged (device, mixed_precision, gradient_accumulation_steps)
```

### TS-005: Real-Time Metrics Logging
```
Given: Training run with 10 epochs
When: Each epoch completes with train_loss and val_loss
Then: W&B dashboard updates within 5 seconds
  And: Loss curves show all 10 data points
  And: X-axis labeled correctly as "Epoch"
  And: Can view metrics while training is still running
```

### TS-006: Multi-Run Comparison
```
Given: User trains same model with 3 different learning rates
When: All 3 runs complete
Then: W&B dashboard shows all runs in same project
  And: Can select multiple runs for side-by-side comparison
  And: Run names distinguish configurations (timestamps + LR in tags)
  And: Best run clearly identifiable by final val_loss
```

## Scenario Category: Error Handling

### TS-007: API Key Exposure Prevention
```
Given: User hardcodes WANDB_API_KEY in notebook
When: They attempt to share notebook or commit to git
Then: Warning cell explains security risk
  And: Notebook includes .gitignore for .wandb/
  And: Instructions to use Colab Secrets or environment variables
```

### TS-008: Network Failure During Training
```
Given: Training in progress with W&B enabled
When: Internet connection lost at epoch 5
Then: Warning logged: "⚠️ W&B sync failed, continuing training"
  And: Training does NOT crash or stop
  And: Epochs 6-10 continue normally
  And: Metrics cached locally in .wandb/
  And: Auto-sync resumes when connectivity restored
```

### TS-009: W&B Service Downtime
```
Given: wandb.ai service temporarily unavailable
When: User attempts to start training
Then: Fallback to offline mode automatically
  And: Clear message: "W&B service unreachable, using offline mode"
  And: Training proceeds without delays
```

## Scenario Category: Integration with Other Features

### TS-010: W&B + Checkpoint Resume
```
Given: Training interrupted at epoch 7 (session timeout)
When: User resumes from checkpoint
Then: New W&B run created with tag "resumed"
  And: Original run ID linked in notes
  And: Metrics continue from epoch 8 (not reset to 0)
  And: Chart shows discontinuity at resume point
```

### TS-011: W&B + HuggingFace Hub Upload
```
Given: Training completes successfully
When: Model pushed to HuggingFace Hub
Then: W&B run metadata includes HF Hub URL
  And: HF model card includes W&B run link
  And: Bidirectional traceability established
```

### TS-012: W&B Artifacts for Checkpoints
```
Given: Training with checkpointing enabled
When: Best model saved at epoch 8
Then: Checkpoint uploaded as W&B artifact
  And: Artifact tagged with metrics (val_loss=2.3)
  And: Can download checkpoint from W&B dashboard
  And: Artifact linked to specific run
```

## Scenario Category: Advanced Features

### TS-013: Custom Charts in Dashboard
```
Given: Training generates perplexity, accuracy, and gradient norms
When: Viewing W&B dashboard
Then: Can create custom chart comparing train_ppl vs val_ppl
  And: Can create panel showing all metrics (loss, ppl, acc) together
  And: Charts auto-update as training progresses
```

### TS-014: Hyperparameter Sweep Integration
```
Given: User runs Optuna hyperparameter search with 20 trials
When: Each trial logs to W&B
Then: All 20 trials visible in same project
  And: Can filter/sort by final val_loss
  And: Parallel coordinates plot shows hyperparameter relationships
  And: Best trial clearly highlighted
```

### TS-015: Alerts for Training Issues
```
Given: Training with W&B alerts configured
When: Gradient norm exceeds 10.0 (potential instability)
Then: W&B sends alert via email/Slack
  And: Alert includes run URL and problematic epoch
  And: User can stop training remotely if needed
```
