import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UnsupervisedActivationsWrapper } from './unsupervised-activations-wrapper';

describe('UnsupervisedActivationsWrapper', () => {
  let component: UnsupervisedActivationsWrapper;
  let fixture: ComponentFixture<UnsupervisedActivationsWrapper>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UnsupervisedActivationsWrapper]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UnsupervisedActivationsWrapper);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
