import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DialogAllHeatmaps } from './dialog-all-heatmaps';

describe('DialogAllHeatmaps', () => {
  let component: DialogAllHeatmaps;
  let fixture: ComponentFixture<DialogAllHeatmaps>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DialogAllHeatmaps]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DialogAllHeatmaps);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
