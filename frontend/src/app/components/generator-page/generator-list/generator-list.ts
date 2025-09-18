import { Component, input } from '@angular/core';

@Component({
  selector: 'app-generator-list',
  imports: [],
  templateUrl: './generator-list.html',
  styleUrl: './generator-list.css',
})
export class GeneratorList {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
}
