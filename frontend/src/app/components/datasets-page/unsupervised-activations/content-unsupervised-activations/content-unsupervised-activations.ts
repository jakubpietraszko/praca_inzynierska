import { Component, input } from '@angular/core';

@Component({
  selector: 'app-content-unsupervised-activations',
  imports: [],
  templateUrl: './content-unsupervised-activations.html',
  styleUrl: './content-unsupervised-activations.css'
})
export class ContentUnsupervisedActivations {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
}
