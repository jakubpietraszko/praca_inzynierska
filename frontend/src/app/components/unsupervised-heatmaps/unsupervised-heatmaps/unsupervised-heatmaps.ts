import { Component, signal } from '@angular/core';

@Component({
  selector: 'app-unsupervised-heatmaps',
  imports: [],
  templateUrl: './unsupervised-heatmaps.html',
  styleUrl: './unsupervised-heatmaps.css'
})
export class UnsupervisedHeatmaps {
  model = signal<string | null>(null);
  heatmap = signal<string | null>(null);
  dataset = signal<string | null>(null);
  label = signal<string | null>(null);


  
}
