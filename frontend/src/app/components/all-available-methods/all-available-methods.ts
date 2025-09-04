import { Component, input } from '@angular/core';

@Component({
  selector: 'app-all-available-methods',
  imports: [],
  templateUrl: './all-available-methods.html',
  styleUrl: './all-available-methods.css'
})
export class AllAvailableMethods {
  model = input<string>();
  dataset = input<string>();
  label = input<string>();
  filename = input<string>();
  target = input<string>();

  
}
