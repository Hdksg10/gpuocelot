; Code assembled by Ocelot LLVMKernel TODO


%LLVMContext = type { %Dimension, %Dimension, %Dimension, %Dimension, i8*, i8*, i8*, i8*, i8*, i8*, i32 };
declare default i32 @__ocelot_get_extent( %LLVMContext* , i32  ) align 1;
declare default float @llvm.pow.f32( float , float  ) align 1;
declare default float @llvm.exp2.f32( float  ) align 1;
declare default float @llvm.log2.f32( float  ) align 1;
declare default float @llvm.log.f32( float  ) align 1;
declare default float @llvm.sin.f32( float  ) align 1;
declare default float @llvm.cos.f32( float  ) align 1;
declare default double @llvm.sqrt.f64( double  ) align 1;
declare default float @llvm.sqrt.f32( float  ) align 1;
declare default i64 @llvm.ctlz.i64( i64  ) align 1;
declare default i32 @llvm.ctlz.i32( i32  ) align 1;
declare default i16 @llvm.ctlz.i16( i16  ) align 1;
declare default i8 @llvm.ctlz.i8( i8  ) align 1;
declare default i64 @llvm.readcyclecounter(  ) align 1;
declare default i64 @llvm.ctpop.i64( i64  ) align 1;
declare default i32 @llvm.ctpop.i32( i32  ) align 1;
declare default i16 @llvm.ctpop.i16( i16  ) align 1;
declare default i8 @llvm.ctpop.i8( i8  ) align 1;
declare default float @nearbyintf( float  ) align 1;
declare default float @truncf( float  ) align 1;
declare default float @ceilf( float  ) align 1;
declare default float @floorf( float  ) align 1;
declare default double @nearbyint( double  ) align 1;
declare default double @trunc( double  ) align 1;
declare default double @ceil( double  ) align 1;
declare default double @floor( double  ) align 1;
declare default i64 @__ocelot_mul_hi_s64( i64 , i64  ) align 1;
declare default i64 @__ocelot_mul_hi_u64( i64 , i64  ) align 1;
declare default i32* @__ocelot_txq( %LLVMContext* , i32 , i32  ) align 1;
declare default i32 @__ocelot_atomic_dec_32( i64 , i32  ) align 1;
declare default i32 @__ocelot_atomic_inc_32( i64 , i32  ) align 1;
declare default void @__ocelot_tex_3d_fs( float* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_3d_fu( float* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_3d_ff( float* , %LLVMContext* , i32 , float , float , float , float  ) align 1;
declare default void @__ocelot_tex_3d_sf( i32* , %LLVMContext* , i32 , float , float , float , float  ) align 1;
declare default void @__ocelot_tex_3d_uf( i32* , %LLVMContext* , i32 , float , float , float , float  ) align 1;
declare default void @__ocelot_tex_3d_ss( i32* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_3d_su( i32* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_3d_us( i32* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_3d_uu( i32* , %LLVMContext* , i32 , i32 , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_2d_fs( float* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_2d_fu( float* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_a2d_ff( float* , %LLVMContext* , i32 , float , float , i32  ) align 1;
declare default void @__ocelot_tex_2d_ff( float* , %LLVMContext* , i32 , float , float  ) align 1;
declare default void @__ocelot_tex_2d_sf( i32* , %LLVMContext* , i32 , float , float  ) align 1;
declare default void @__ocelot_tex_2d_uf( i32* , %LLVMContext* , i32 , float , float  ) align 1;
declare default void @__ocelot_tex_2d_ss( i32* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_2d_su( i32* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_2d_us( i32* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_2d_uu( i32* , %LLVMContext* , i32 , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_fs( float* , %LLVMContext* , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_fu( float* , %LLVMContext* , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_ff( float* , %LLVMContext* , i32 , float  ) align 1;
declare default void @__ocelot_tex_1d_sf( i32* , %LLVMContext* , i32 , float  ) align 1;
declare default void @__ocelot_tex_1d_uf( i32* , %LLVMContext* , i32 , float  ) align 1;
declare default void @__ocelot_tex_1d_ss( i32* , %LLVMContext* , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_su( i32* , %LLVMContext* , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_us( i32* , %LLVMContext* , i32 , i32  ) align 1;
declare default void @__ocelot_tex_1d_uu( i32* , %LLVMContext* , i32 , i32  ) align 1;
declare default i1 @__ocelot_vote( i1 , i32 , i1  ) align 1;
declare default i32 @__ocelot_prmt_rc16( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt_ecr( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt_ecl( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt_rc8( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt_b4e( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt_f4e( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_prmt( i32 , i32 , i32  ) align 1;
declare default i32 @__ocelot_bfind_b64( i64 , i1  ) align 1;
declare default i32 @__ocelot_bfind_b32( i32 , i1  ) align 1;
declare default i64 @__ocelot_bfi_b64( i64 , i64 , i32 , i32  ) align 1;
declare default i32 @__ocelot_bfi_b32( i32 , i32 , i32 , i32  ) align 1;
declare default i64 @__ocelot_bfe_b64( i64 , i32 , i32 , i1  ) align 1;
declare default i32 @__ocelot_bfe_b32( i32 , i32 , i32 , i1  ) align 1;
declare default i64 @__ocelot_brev_b64( i64  ) align 1;
declare default i32 @__ocelot_brev_b32( i32  ) align 1;

%Dimension = type { i32, i32, i32 };
define default void @_Z_ocelotTranslated__Z9test_lop3PjS_S_S_( %LLVMContext* %__ctaContext ) nounwind align 1;
{
BB_1_2:
	%rt0 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 7;
	%rt1 = load i8*, i8** %rt0;
	%rt2 = bitcast i8* %rt1 to i64*;
	%r0 = load i64, i64* %rt2, align 8;
	%rt3 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 7;
	%rt4 = load i8*, i8** %rt3;
	%rt5 = bitcast i8* %rt4 to i64*;
	%r1 = load i64, i64* %rt5, align 8;
	%rt6 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 7;
	%rt7 = load i8*, i8** %rt6;
	%rt8 = bitcast i8* %rt7 to i64*;
	%r2 = load i64, i64* %rt8, align 8;
	%rt9 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 7;
	%rt10 = load i8*, i8** %rt9;
	%rt11 = bitcast i8* %rt10 to i64*;
	%r3 = load i64, i64* %rt11, align 8;
	%r4 = bitcast i64 %r3 to i64;
	%r5 = bitcast i64 %r2 to i64;
	%r6 = bitcast i64 %r1 to i64;
	%r7 = bitcast i64 %r0 to i64;
	%rt12 = inttoptr i64 %r7 to i32*;
	%r8 = load i32, i32* %rt12, align 4;
	%rt13 = inttoptr i64 %r6 to i32*;
	%r9 = load i32, i32* %rt13, align 4;
	%rt14 = inttoptr i64 %r5 to i32*;
	%r10 = load i32, i32* %rt14, align 4;
	%rt15 = alloca i32;
	%rt16 = alloca i32;
	store i32 0, i32* %rt15;
	store i32 0, i32* %rt16;
	br label %Ocelot_cond_lop3_BB_1_2;
Ocelot_cond_lop3_BB_1_2:
	%rt17 = load i32, i32* %rt16;
	%rt18 = icmp slt i32 %rt17, 32;
	br i1 %rt18, label %Ocelot_loop_lop3_BB_1_2, label %Ocelot_end_lop3_BB_1_2;
Ocelot_loop_lop3_BB_1_2:
	%rt19 = load i32, i32* %rt15;
	%rt20 = lshr i32 %r8, %rt17;
	%rt21 = lshr i32 %r9, %rt17;
	%rt22 = lshr i32 %r10, %rt17;
	%rt23 = and i32 %rt20, 1;
	%rt24 = and i32 %rt21, 1;
	%rt25 = and i32 %rt22, 1;
	%rt26 = shl i32 %rt23, 2;
	%rt27 = shl i32 %rt24, 1;
	%rt28 = or i32 %rt26, %rt27;
	%rt29 = or i32 %rt28, %rt25;
	%rt30 = trunc i32 %rt29 to i8;
	%rt31 = lshr i8 -1, %rt30;
	%rt32 = and i8 %rt31, 1;
	%rt33 = zext i8 %rt31 to i32;
	%rt34 = shl i32 %rt33, %rt17;
	%rt35 = or i32 %rt19, %rt34;
	store i32 %rt35, i32* %rt15;
	%rt36 = add i32 %rt17, 1;
	store i32 %rt36, i32* %rt16;
	br label %Ocelot_cond_lop3_BB_1_2;
Ocelot_end_lop3_BB_1_2:
	%r11 = load i32, i32* %rt15;
	%rt37 = inttoptr i64 %r4 to i32*;
	store i32 %r11, i32* %rt37, align 4;
	%rt38 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 4;
	%rt39 = load i8*, i8** %rt38;
	%rt40 = bitcast i8* %rt39 to i32*;
	store i32 2, i32* %rt40;
	br label %BB_1_1;
BB_1_1:
	ret void;

}

