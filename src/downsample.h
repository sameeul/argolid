#pragma once

#include<vector>
#include<memory>
#include<cmath>
// helper function to downsample as an average
template <typename T>
std::unique_ptr<std::vector<T>>DownsampleAverage(std::vector<T>& source_array, std::int64_t row, std::int64_t col) {
  auto new_row = static_cast<std::int64_t>(ceil(row / 2.0));
  auto new_col = static_cast<std::int64_t>(ceil(col / 2.0));
  auto result_data = std::make_unique<std::vector<T>>(new_row * new_col);
  auto data_ptr = result_data->data();

  int even_row{0}, even_col{0};
  
  if (row%2==0){
    even_row = row;
  }
  else {
    even_row = row-1;
  }

  if (col%2==0){
    even_col = col;
  }
  else {
    even_col = col-1;
  }

  for (std::int64_t i = 0; i < even_row; i=i+2) {
    std::int64_t row_offset = (i / 2) * new_col;
    std::int64_t prev_row_offset = i * col;
    std::int64_t prev_row_offset_2 = (i+1) * col;
    for (std::int64_t j = 0; j < even_col; j = j + 2) {
      std::int64_t new_data_index = row_offset + (j / 2);
      data_ptr[new_data_index] =   (source_array[prev_row_offset + j] 
                                   + source_array[prev_row_offset + j + 1]
                                   + source_array[prev_row_offset_2 + j]
                                   + source_array[prev_row_offset_2 + j + 1])*0.25;
    }
  }
  // fix the last col if odd
  if (col % 2 == 1) {
    for (std::int64_t i = 0; i < even_row; i=i+2) {
      data_ptr[((i / 2)+1) * new_col-1] = 0.5*(source_array[(i+1)*col-1] + source_array[(i+2)*col-1]);
    }
  }

  // fix the last row if odd
  if (row % 2 == 1) {
    std::int64_t col_offset = (new_row-1)*new_col;
    std::int64_t old_col_offset =(row-1)*col;
    for (std::int64_t i = 0; i < even_col; i=i+2) {
      data_ptr[col_offset+(i/2)] = 0.5*(source_array[old_col_offset+i] + source_array[old_col_offset+i+1]);
    }
  }
  
  // fix the last element if both row and col are odd
  if (row%2==1 && col%2==1){
      data_ptr[new_row*new_col-1] = source_array[row*col-1];
  }

  return result_data;
}

template <typename T>
std::unique_ptr<std::vector<T>>DownsampleModeMin(std::vector<T>& source_array, std::int64_t row, std::int64_t col) {
  auto new_row = static_cast<std::int64_t>(ceil(row / 2.0));
  auto new_col = static_cast<std::int64_t>(ceil(col / 2.0));
  auto result_data = std::make_unique<std::vector<T>>(new_row * new_col);
  auto data_ptr = result_data->data();

  int even_row{0}, even_col{0};
  
  if (row%2==0){
    even_row = row;
  }
  else {
    even_row = row-1;
  }

  if (col%2==0){
    even_col = col;
  }
  else {
    even_col = col-1;
  }

  for (std::int64_t i = 0; i < even_row; i=i+2) {
    std::int64_t row_offset = (i / 2) * new_col;
    std::int64_t prev_row_offset = i * col;
    std::int64_t prev_row_offset_2 = (i+1) * col;
    for (std::int64_t j = 0; j < even_col; j = j + 2) {
      std::int64_t new_data_index = row_offset + (j / 2);

      // if no number is same
      if( (source_array[prev_row_offset + j] == source_array[prev_row_offset + j + 1]) ||
          (source_array[prev_row_offset + j] == source_array[prev_row_offset_2 + j]) ||
          (source_array[prev_row_offset + j] == source_array[prev_row_offset_2 + j + 1])){
            data_ptr[new_data_index] = source_array[prev_row_offset + j];
          }

      else if((source_array[prev_row_offset + j + 1] == source_array[prev_row_offset_2 + j]) ||
              (source_array[prev_row_offset + j + 1] == source_array[prev_row_offset_2 + j + 1])){
            data_ptr[new_data_index] = source_array[prev_row_offset + j + 1];
          }

      else {
        data_ptr[new_data_index] = std::min({source_array[prev_row_offset + j], source_array[prev_row_offset + j + 1],
                                            source_array[prev_row_offset_2 + j], source_array[prev_row_offset_2 + j + 1]});
      }          

    }
  }
  // fix the last col if odd
  if (col % 2 == 1) {
    for (std::int64_t i = 0; i < even_row; i=i+2) {
      data_ptr[((i / 2)+1) * new_col-1] = std::min({(source_array[(i+1)*col-1], source_array[(i+2)*col-1])});
    }
  }

  // fix the last row if odd
  if (row % 2 == 1) {
    std::int64_t col_offset = (new_row-1)*new_col;
    std::int64_t old_col_offset =(row-1)*col;
    for (std::int64_t i = 0; i < even_col; i=i+2) {
      data_ptr[col_offset+(i/2)] = std::min({source_array[old_col_offset+i], source_array[old_col_offset+i+1]});
    }
  }
  
  // fix the last element if both row and col are odd
  if (row%2==1 && col%2==1){
      data_ptr[new_row*new_col-1] = source_array[row*col-1];
  }

  return result_data;
}

template <typename T>
std::unique_ptr<std::vector<T>>DownsampleModeMax(std::vector<T>& source_array, std::int64_t row, std::int64_t col) {
  auto new_row = static_cast<std::int64_t>(ceil(row / 2.0));
  auto new_col = static_cast<std::int64_t>(ceil(col / 2.0));
  auto result_data = std::make_unique<std::vector<T>>(new_row * new_col);
  auto data_ptr = result_data->data();

  int even_row{0}, even_col{0};
  
  if (row%2==0){
    even_row = row;
  }
  else {
    even_row = row-1;
  }

  if (col%2==0){
    even_col = col;
  }
  else {
    even_col = col-1;
  }

  for (std::int64_t i = 0; i < even_row; i=i+2) {
    std::int64_t row_offset = (i / 2) * new_col;
    std::int64_t prev_row_offset = i * col;
    std::int64_t prev_row_offset_2 = (i+1) * col;
    for (std::int64_t j = 0; j < even_col; j = j + 2) {
      std::int64_t new_data_index = row_offset + (j / 2);

      // if no number is same
      if( (source_array[prev_row_offset + j] == source_array[prev_row_offset + j + 1]) ||
          (source_array[prev_row_offset + j] == source_array[prev_row_offset_2 + j]) ||
          (source_array[prev_row_offset + j] == source_array[prev_row_offset_2 + j + 1])){
            data_ptr[new_data_index] = source_array[prev_row_offset + j];
          }

      else if((source_array[prev_row_offset + j + 1] == source_array[prev_row_offset_2 + j]) ||
              (source_array[prev_row_offset + j + 1] == source_array[prev_row_offset_2 + j + 1])){
            data_ptr[new_data_index] = source_array[prev_row_offset + j + 1];
          }

      else {
        data_ptr[new_data_index] = std::max({source_array[prev_row_offset + j], source_array[prev_row_offset + j + 1],
                                            source_array[prev_row_offset_2 + j], source_array[prev_row_offset_2 + j + 1]});
      }          

    }
  }
  // fix the last col if odd
  if (col % 2 == 1) {
    for (std::int64_t i = 0; i < even_row; i=i+2) {
      data_ptr[((i / 2)+1) * new_col-1] = std::max({(source_array[(i+1)*col-1], source_array[(i+2)*col-1])});
    }
  }

  // fix the last row if odd
  if (row % 2 == 1) {
    std::int64_t col_offset = (new_row-1)*new_col;
    std::int64_t old_col_offset =(row-1)*col;
    for (std::int64_t i = 0; i < even_col; i=i+2) {
      data_ptr[col_offset+(i/2)] = std::max({source_array[old_col_offset+i], source_array[old_col_offset+i+1]});
    }
  }
  
  // fix the last element if both row and col are odd
  if (row%2==1 && col%2==1){
      data_ptr[new_row*new_col-1] = source_array[row*col-1];
  }

  return result_data;
}
