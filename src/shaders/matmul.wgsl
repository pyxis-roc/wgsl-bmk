const WORKGROUP_SIZE_X = ${WORKGROUP_SIZE_X};
const WORKGROUP_SIZE_Y = ${WORKGROUP_SIZE_Y};
const WORK_PER_THREAD: vec2<i32> = vec2<i32>(${WORK_PER_THREAD_X}, ${WORK_PER_THREAD_Y});

const RowPerThread: i32 = WORK_PER_THREAD.y;
const ColPerThread: i32 = WORK_PER_THREAD.x;

const TileAOuter: i32 = WORKGROUP_SIZE_Y * RowPerThread;
const TileBOuter: i32 = WORKGROUP_SIZE_Y * ColPerThread;
const TileInner: i32 = max(TileAOuter, TileBOuter);


@group(0) @binding(0) var<storage> firstMatrix: array<f32>;
@group(0) @binding(1) var<storage> secondMatrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultMatrix: array<f32>;
@group(0) @binding(3) var<uniform> iShape: vec4<i32>;

var<workgroup> mm_Asub: array<array<f32, TileAOuter>, TileInner>;
var<workgroup> mm_Bsub: array<array<f32, TileInner>, TileBOuter>;
var<private> gl_LocalInvocationID_1: vec3<u32>;
var<private> gl_GlobalInvocationID_1: vec3<u32>;

fn mm_readA(row: i32, col: i32) -> f32 {
    return firstMatrix[row * iShape[1] + col];
}

fn mm_readB(row: i32, col: i32) -> f32 {
    return secondMatrix[row *  iShape[3] + col];
}

fn mm_write(row: i32, col: i32, value: f32) {
    resultMatrix[row * iShape[3] + col] = value;
}

fn mm_matMul(local_invocation_id: vec3<u32>, global_invocation_id: vec3<u32>) {
  var ACached: f32;
  var BCached: array<f32, ColPerThread>;

  let dimAOuter = iShape[0];
  let dimInner = iShape[1];
  let dimBOuter = iShape[3];
  let tileRow = i32(local_invocation_id.y) * RowPerThread;
  let tileCol = i32(local_invocation_id.x) * ColPerThread;
  
  let globalRow = i32(global_invocation_id.y) * RowPerThread;
  let globalCol = i32(global_invocation_id.x) * ColPerThread;

  let numTiles = (dimInner - 1) / TileInner + 1;


  var acc: array<array<f32, 64>, 64>;

  for(var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
    for(var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
      acc[innerRow][innerCol] = 0.0;
    }
  }

  let ColPerThreadA: i32 = (TileInner / WORKGROUP_SIZE_X);
  var tileColA = i32(local_invocation_id.x) * ColPerThreadA;
  let RowPerThreadB: i32 = TileInner / WORKGROUP_SIZE_Y;
  var tileRowB = i32(local_invocation_id.y) * RowPerThreadB;

  // loop over shared dimension
  for (var t = i32(0); t < numTiles; t++) {
    // Load one tile of A into workgroup memory
    for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThreadA; innerCol++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileColA + innerCol;
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, t * TileInner + inputCol);
        }
    }
    // Load one tile of B into workgroup memory
    for (var innerRow = i32(0); innerRow < RowPerThreadB; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol + innerCol);
        }
    }

    workgroupBarrier();

    // Compute acc values for a single thread.
    for (var k = i32(0); k < TileInner; k++) {
        for (var inner = i32(0); inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
        }

        for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
                acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
        }
    }

    workgroupBarrier();

    for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
            if ((globalCol + innerCol) < dimBOuter && (globalRow + innerRow) < dimAOuter) {
                mm_write(globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);
            }
        }
    }

  }

}

@compute @workgroup_size(64, 64, 1) 
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    mm_matMul(local_id, gid);
    return;
}
