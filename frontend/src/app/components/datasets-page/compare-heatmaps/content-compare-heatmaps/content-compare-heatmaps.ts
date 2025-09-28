import { Component, input } from '@angular/core';

@Component({
  selector: 'app-content-compare-heatmaps',
  imports: [],
  templateUrl: './content-compare-heatmaps.html',
  styleUrl: './content-compare-heatmaps.css'
})
export class ContentCompareHeatmaps {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>()
  filename = input.required<string>();
}
