import { Component, inject, input, OnDestroy, OnInit } from '@angular/core';
import { HeatmapsWebsocket } from '../../../services/heatmaps-websocket';

@Component({
  selector: 'app-generator-list',
  imports: [],
  templateUrl: './generator-list.html',
  styleUrl: './generator-list.css',
})
export class GeneratorList implements OnInit, OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();

  ws = inject(HeatmapsWebsocket);

  ngOnInit() {
    this.ws.connect();
  }

  ngOnDestroy() {
    this.ws.disconnect();
  }
}
