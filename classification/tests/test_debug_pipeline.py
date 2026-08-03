import os
import shutil
import tempfile
import torch
import numpy as np
from models.DAMamba import DAMamba
from debug.recorder import get_dcnv3_grid, get_approximate_grid
from debug.metrics import compute_offset_metrics, compute_feature_metrics
from debug.export import export_debug_data
from debug.visualizer import DASVisualizer
from models.DAMamba import DAMamba, Dynamic_Adaptive_Scan

def test_debug_pipeline():
    # 1. Instantiate a small model for testing
    model = DAMamba(
        in_chans=3,
        num_classes=5,
        depths=(2, 2),
        dims=(16, 32),
        mlp_ratios=(4, 4),
        drop_path_rate=0.1,
        layerscale=(False, False),
        use_attention=False
    )
    
    # Create a random input
    x = torch.rand(1, 3, 32, 32)
    
    # 2. Test normal forward pass (debug disabled by default)
    model.eval()
    with torch.no_grad():
        out_normal = model(x)
        
    # Check that debug_data is empty
    for m in model.modules():
       if isinstance(m, Dynamic_Adaptive_Scan):
            assert len(m.debug_data) == 0
            assert not m.debug
            
    # 3. Enable debug mode
    model.enable_scan_debug()
    
    # Check that debug states are propagated
    assert model.debug
    for m in model.modules():
       if isinstance(m, Dynamic_Adaptive_Scan):
            assert m.debug
            
    # Run forward pass under debug
    with torch.no_grad():
        out_debug = model(x)
        
    # 4. Numerical verification: debug mode must NOT alter inference outputs
    assert torch.allclose(out_normal, out_debug, rtol=1e-5, atol=1e-5)
    
    # Retrieve debug records
    records = model.get_scan_debug_data()
    assert len(records) > 0
    
    # Verify the structure of the first record
    first_record = records[0]
    required_keys = [
        "stage", "block", "H", "W", "kernel_size", "group", "offset_scale",
        "input", "dw_conv_out", "raw_offset", "base_grid", "sampling_grid",
        "base_grid_normalized", "sampling_grid_normalized", "approx_grid",
        "offset_magnitude", "offset_metrics", "feature_metrics", "scanned",
        "selective_scan_input"
    ]
    for key in required_keys:
        assert key in first_record, f"Key {key} missing from debug record"
        
    # Verify shape consistency
    H, W = first_record["H"], first_record["W"]
    group = first_record["group"]
    num_points = first_record["kernel_size"] * first_record["kernel_size"]
    
    assert first_record["raw_offset"].shape == (H, W, group, num_points, 2)
    assert first_record["base_grid"].shape == (H, W, group, num_points, 2)
    assert first_record["sampling_grid"].shape == (H, W, group, num_points, 2)
    assert first_record["approx_grid"].shape == (H, W, group, num_points, 2)
    
    # 5. Verify visualizer & exporter
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock image
        mock_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Test Export
        meta_extra = {
            "predicted_class": 2,
            "confidence": 0.85,
            "true_label": 2,
            "correct": True,
            "input_image_size": (32, 32)
        }
        export_debug_data(temp_dir, "test_img", records, meta_extra)
        
        # Check files were written
        img_dir = os.path.join(temp_dir, "test_img")
        assert os.path.exists(img_dir)
        assert os.path.exists(os.path.join(img_dir, "metadata.json"))
        
        stage_dir = os.path.join(img_dir, "stage_0_block_0")
        assert os.path.exists(stage_dir)
        assert os.path.exists(os.path.join(stage_dir, "offset.npy"))
        assert os.path.exists(os.path.join(stage_dir, "grid.npy"))
        assert os.path.exists(os.path.join(stage_dir, "metadata.json"))
        
        # Test Visualizer
        viz = DASVisualizer(temp_dir)
        viz.save(mock_img, "test_img", records)
        
        # Check plots were saved
        expected_plots = [
            "scan_vectors.png", "offset_heatmap.png", "sampling_grid.png",
            "feature_before_after.png", "offset_histogram.png"
        ]
        for plot in expected_plots:
            assert os.path.exists(os.path.join(stage_dir, plot)), f"Missing plot: {plot}"
            
    finally:
        shutil.rmtree(temp_dir)
        
    # Test disable & clear scan debug
    model.clear_scan_debug()
    assert len(model.get_scan_debug_data()) == 0
    
    model.disable_scan_debug()
    for m in model.modules():
       if isinstance(m, Dynamic_Adaptive_Scan):
            assert not m.debug
