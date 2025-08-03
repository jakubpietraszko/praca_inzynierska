import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ImageDirective } from '../../directives/image/image-directive';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-image-component',
  imports: [CommonModule, ImageDirective],
  template: `
    <div class="image-container">
      <h2>Dataset Image {{name}}</h2>
      @for(item of imageArray; track item){
        <img
          appImageDirective
          [imagePath]="'mnist-dataset/five/8.png'"
        />
      }
    </div>
  `,
  styleUrl: './image-component.css'
})
export class ImageComponent {
  imageArray: number[] = Array(1).fill(0).map((_, i) => i + 1);
  name: string = environment.title;

  constructor() {}

}