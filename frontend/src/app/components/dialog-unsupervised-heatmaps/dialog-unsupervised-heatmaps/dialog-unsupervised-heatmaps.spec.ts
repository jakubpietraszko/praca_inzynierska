import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DialogUnsupervisedHeatmaps } from './dialog-unsupervised-heatmaps';

describe('DialogUnsupervisedHeatmaps', () => {
  let component: DialogUnsupervisedHeatmaps;
  let fixture: ComponentFixture<DialogUnsupervisedHeatmaps>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DialogUnsupervisedHeatmaps]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DialogUnsupervisedHeatmaps);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
