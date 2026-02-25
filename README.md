<h1 align="center">YOLOv8 Object Tracking and Robustness Analysis</h1>

<hr>

<h2>Overview</h2>

<p>
This project implements a full computer vision pipeline that performs object detection 
and multi-object tracking on videos using <b>YOLOv8</b>. 
It further evaluates system robustness by testing performance under various 
challenging visual conditions such as noise, blur, low brightness, occlusion, and frame dropping.
</p>

<p>
The system measures both inference speed and tracking quality degradation, 
making it suitable for performance benchmarking and experimental analysis.
</p>

<hr>

<h2>Key Features</h2>

<ul>
  <li>YOLOv8-based object detection</li>
  <li>Multi-object tracking with persistent unique IDs</li>
  <li>Annotated output videos with bounding boxes and labels</li>
  <li>Robustness testing under:
    <ul>
      <li>Gaussian Noise</li>
      <li>Motion Blur</li>
      <li>Low Brightness</li>
      <li>Occlusion Patches</li>
      <li>Frame Dropping</li>
    </ul>
  </li>
  <li>Automatic JSON result logging</li>
  <li>Automated performance comparison plots</li>
</ul>

<hr>

<h2>Evaluation Metrics</h2>

<ul>
  <li><b>Average FPS</b> – Processing speed</li>
  <li><b>Average Latency (ms)</b> – Inference time per frame</li>
  <li><b>Unique Objects Tracked</b> – Tracking stability</li>
  <li><b>FPS Drop (%)</b> – Speed degradation vs clean input</li>
  <li><b>Tracking Quality Drop (%)</b> – Tracking degradation vs clean baseline</li>
</ul>

<hr>

