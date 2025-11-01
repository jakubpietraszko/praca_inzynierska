import { ChangeDetectionStrategy, Component, input, TemplateRef } from '@angular/core';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { NgTemplateOutlet } from '@angular/common';

export interface UnsupervisedCoordinate {
  coordX: number;
  coordY: number;
  label: string;
  filename: string;
}

@Component({
  selector: 'app-generic-unsupervised',
  imports: [DragDropModule, NgTemplateOutlet],
  templateUrl: './generic-unsupervised.html',
  styleUrl: './generic-unsupervised.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericUnsupervised {
  coordinates = input.required<UnsupervisedCoordinate[]>();
  imageTemplate = input.required<TemplateRef<any>>();
}
