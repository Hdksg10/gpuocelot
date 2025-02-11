; Code assembled by Ocelot LLVMKernel TODO


%LLVMContext = type { %Dimension, %Dimension, %Dimension, %Dimension, i8*, i8*, i8*, i8*, i8*, i8*, i32 };
declare default i32 @__ocelot_get_extent( %LLVMContext* , i32  ) align 1;
declare default i16 @llvm.convert.to.fp16.f32( float  ) align 1;
declare default float @llvm.convert.from.fp16.f32( i16  ) align 1;
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
define default void @_Z_ocelotTranslated__Z20test_setp_f16_boolopPi( %LLVMContext* %__ctaContext ) nounwind align 1;
{
$OcelotRegisterInitializerBlock:
	%ri0 = bitcast i32 0 to i32;
	br label %BB_3_2;
BB_3_2:
	%r0 = phi i32 [ %ri0, %$OcelotRegisterInitializerBlock ];
	%rt0 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 7;
	%rt1 = load i8*, i8** %rt0;
	%rt2 = bitcast i8* %rt1 to i64*;
	%r1 = load i64, i64* %rt2, align 8;
	%r2 = bitcast i64 %r1 to i64;
	%r3 = bitcast i16 15360 to i16;
	%r4 = icmp eq i32 1, 34;
	%rt4 = bitcast i16 %r3 to half;
	%rt5 = bitcast i16 %r3 to half;
	%rt3 = fcmp oeq half %rt4, %rt5;
	%r5 = and i1 %r4, %rt3;
	%r6 = bitcast i32 1 to i32;
	%r7 = select i1 %r5, i32 %r6, i32 %r0;
	%rt6 = inttoptr i64 %r2 to i32*;
	store i32 %r7, i32* %rt6, align 4;
	%rt7 = getelementptr %LLVMContext, %LLVMContext* %__ctaContext, i32 0, i32 4;
	%rt8 = load i8*, i8** %rt7;
	%rt9 = bitcast i8* %rt8 to i32*;
	store i32 2, i32* %rt9;
	br label %BB_3_1;
BB_3_1:
	ret void;

}

