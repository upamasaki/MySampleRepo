@echo off
rem �t�@�C�����̈ꗗ���擾

rem ==================================
echo "+++++ �m�F���� +++++"
echo "+"
echo "+ OK, NG, AnnotationMiss�̃t�H���_�͂���܂����H"
echo "+ �摜��windows�Őݒ肳��Ă���A�v�����N�����܂�"
echo "+ (�G���[����������ꍇ�͊Ǘ��҂Ŏ��s���Ă݂Ă�������)"
echo "+"
echo "+++++++++++++++"

rem ==================================
rem ���x���ƃL�[�{�[�h�Ƃ̊��蓖��
set LABEL_OK=1
set LABEL_NG=3
set LABEL_AnnotationMiss=5

rem ==================================
echo "��������t�@�C����:"
for  %%A in (*.JPG) do ( if exist %%A (set /a counter=counter+1) )
echo %counter%
set i=0

rem ==================================
rem Main�̏����J�n
SETLOCAL enabledelayedexpansion
for %%A in (*.JPG) do (
	
	echo "=============================="
	echo "���݂̐i��:%i% / %counter%"
	rem ==================================
	rem �摜�̕`��
	echo "�t�@�C����:"
        echo %%A
	%%A

	rem ==================================
	REM ���[�U�[����̓��͂��󂯕t����
	set /p INPUT_LABEL="���x������͂��Ă������� \n(%LABEL_OK%:OK, %LABEL_NG%:NG, %LABEL_AnnotationMiss%:AnnotationMiss) >>>"
	echo INPUT_LABEL is !INPUT_LABEL! .

	rem ==================================
	REM �摜�̈ړ�
	if !INPUT_LABEL! == %LABEL_OK% (
		echo INPUT_LABEL is %LABEL_OK% .
		move %%A .\OK
	)

	if !INPUT_LABEL! == %LABEL_NG% (
		echo INPUT_LABEL is %LABEL_NG% .
		move %%A .\NG
	)

	if !INPUT_LABEL! == %LABEL_AnnotationMiss% (
		echo INPUT_LABEL is %LABEL_AnnotationMiss% .
		move %%A .\NG
	)

	rem ==================================
	REM �摜�̕`��I��
	taskkill /F /IM Microsoft.Photos.exe
)
ENDLOCAL
echo "+++++++++++++++"
echo "+"
echo "+ �I���ł��I�I
echo "+ �����l�ł���(���L��`��)�D
echo "+"
echo "+++++++++++++++"
pause