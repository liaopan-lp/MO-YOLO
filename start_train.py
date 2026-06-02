"""
Three-stage training script for DecoderTracker (MO-YOLO).

Stage 1: Pure detection training (no tracking, is_first=True for all frames)
    - Uses standard DETR-style detection loss (cls + l1 + giou + anc)
    - Train the network to extract target image features efficiently
    - Can use larger batch size and data augmentation

Stage 2: TBSP training (Tracking Box Selection Process, weak tracking)
    - Uses MOTR's Collective Average Loss with TBSP filtering
    - TBSP filters detect queries that overlap with track queries (IOU > threshold)
    - This is a self-supervised/weak supervision pretraining for stage 3
    - Does NOT use TALA (Tracklet-Aware Label Assignment)

Stage 3: TALA training (full tracking with MOTR's training strategy)
    - Uses MOTR's TALA for comprehensive tracking performance
    - TBSP can be disabled after sufficient training in this stage
    - The current code behavior corresponds to this stage
"""

from ultralytics import DecoderTracker
import os


def train_stage_1(data_config, epochs=50, batch=8, imgsz=640, device='', resume=None, 
                  pretrained_weights=None):
    """
    Stage 1: Pure Detection Training
    
    The network is trained as a detection network. Every frame is treated as the first frame
    (is_first=True), so no tracking queries are carried over between frames.
    This allows the network to learn appearance characteristics of objects to be tracked.
    """
    print("=" * 80)
    print("Stage 1: Pure Detection Training")
    print("=" * 80)
    
    model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=1)
    
    if pretrained_weights:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        model.load(pretrained_weights)
    
    # Stage 1: can use larger batch size since no tracking state is maintained
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        resume=resume if resume else False,
        deterministic=False,
        close_mosaic=max(10, epochs // 5),  # close mosaic augmentation in last 20% epochs
    )
    
    print(f"\nStage 1 training complete. Best model saved at: {model.trainer.best}")
    return str(model.trainer.best)


def train_stage_2(data_config, stage1_weights=None, epochs=30, batch=1, imgsz=640, 
                  device='', resume=None):
    """
    Stage 2: TBSP Training (Weak Tracking)
    
    Uses TBSP (Tracking Box Selection Process) as a preprocessing step for TALA.
    - Track queries are maintained across frames (multi-frame tracking)
    - Detect queries that overlap with active track queries (IOU > threshold) are filtered out
    - This prevents detect queries from being assigned to already-tracked objects
    - Acts as self-supervised/weak supervision for the tracking capability
    """
    print("=" * 80)
    print("Stage 2: TBSP Training (Weak Tracking)")
    print("=" * 80)
    
    model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=2)
    
    if stage1_weights:
        print(f"Loading Stage 1 weights from: {stage1_weights}")
        model.load(stage1_weights)
    
    # Stage 2: batch=1 required for video-based training
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        resume=resume if resume else False,
        deterministic=False,
    )
    
    print(f"\nStage 2 training complete. Best model saved at: {model.trainer.best}")
    return str(model.trainer.best)


def train_stage_3(data_config, stage2_weights=None, epochs=30, batch=1, imgsz=640, 
                  device='', resume=None):
    """
    Stage 3: TALA Training (Full Tracking)
    
    Uses MOTR's Tracklet-Aware Label Assignment (TALA) for comprehensive tracking.
    - TBSP is disabled; TALA handles the label assignment
    - After sufficient training in this stage, TBSP can also be disabled during inference
    - This stage fully trains the tracking capability
    """
    print("=" * 80)
    print("Stage 3: TALA Training (Full Tracking)")
    print("=" * 80)
    
    model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=3)
    
    if stage2_weights:
        print(f"Loading Stage 2 weights from: {stage2_weights}")
        model.load(stage2_weights)
    
    # Stage 3: batch=1 required for video-based training
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        resume=resume if resume else False,
        deterministic=False,
    )
    
    print(f"\nStage 3 training complete. Best model saved at: {model.trainer.best}")
    return str(model.trainer.best)


def train_all_stages(data_config, device='', 
                     stage1_epochs=50, stage1_batch=8,
                     stage2_epochs=30, stage3_epochs=30):
    """
    Run all three training stages sequentially.
    
    Args:
        data_config: Path to dataset YAML config (e.g., 'dancetracker.yaml' or 'MOT.yaml')
        device: GPU device (e.g., '0' or '0,1' or '' for auto)
        stage1_epochs: Number of epochs for stage 1
        stage1_batch: Batch size for stage 1 (can be larger)
        stage2_epochs: Number of epochs for stage 2
        stage3_epochs: Number of epochs for stage 3
    """
    print("\n" + "=" * 80)
    print("DecoderTracker Three-Stage Training Pipeline")
    print("=" * 80 + "\n")
    
    # Stage 1: Pure detection
    stage1_best = train_stage_1(
        data_config=data_config,
        epochs=stage1_epochs,
        batch=stage1_batch,
        device=device,
    )
    
    # Stage 2: TBSP (weak tracking)
    stage2_best = train_stage_2(
        data_config=data_config,
        stage1_weights=stage1_best,
        epochs=stage2_epochs,
        device=device,
    )
    
    # Stage 3: TALA (full tracking)
    stage3_best = train_stage_3(
        data_config=data_config,
        stage2_weights=stage2_best,
        epochs=stage3_epochs,
        device=device,
    )
    
    print("\n" + "=" * 80)
    print("All three training stages completed!")
    print(f"Stage 1 best: {stage1_best}")
    print(f"Stage 2 best: {stage2_best}")
    print(f"Stage 3 best: {stage3_best}")
    print("=" * 80)
    
    return stage3_best


if __name__ == '__main__':
    # ============================================================
    # Configuration - Modify these parameters for your training
    # ============================================================
    DATA_CONFIG = "dancetracker.yaml"  # Dataset config: 'dancetracker.yaml' or 'MOT.yaml'
    DEVICE = ''  # GPU device: '0', '0,1', or '' for auto
    
    # Stage 1: Pure Detection
    STAGE1_EPOCHS = 50
    STAGE1_BATCH = 8  # Can use larger batch for pure detection
    
    # Stage 2: TBSP
    STAGE2_EPOCHS = 30
    
    # Stage 3: TALA
    STAGE3_EPOCHS = 30
    
    # ============================================================
    # Training modes - uncomment the one you want to use
    # ============================================================
    
    # Option 1: Run all three stages sequentially
    # train_all_stages(
    #     data_config=DATA_CONFIG,
    #     device=DEVICE,
    #     stage1_epochs=STAGE1_EPOCHS,
    #     stage1_batch=STAGE1_BATCH,
    #     stage2_epochs=STAGE2_EPOCHS,
    #     stage3_epochs=STAGE3_EPOCHS,
    # )
    
    # Option 2: Run individual stages (useful for resuming or debugging)
    
    # Stage 1 only
    # train_stage_1(data_config=DATA_CONFIG, epochs=STAGE1_EPOCHS, batch=STAGE1_BATCH, device=DEVICE)
    
    # Stage 2 only (with pretrained Stage 1 weights)
    # train_stage_2(data_config=DATA_CONFIG, stage1_weights="path/to/stage1_best.pt", epochs=STAGE2_EPOCHS, device=DEVICE)
    
    # Stage 3 only (with pretrained Stage 2 weights)
    # train_stage_3(data_config=DATA_CONFIG, stage2_weights="path/to/stage2_best.pt", epochs=STAGE3_EPOCHS, device=DEVICE)
    
    # Option 3: Quick test with a single stage
    model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=1)
    model.train(data=DATA_CONFIG, epochs=1, batch=1)